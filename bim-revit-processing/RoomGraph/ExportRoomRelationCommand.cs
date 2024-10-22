using System.Diagnostics;
using System.IO;
using System.Windows;
using Autodesk.Revit.Attributes;
using Autodesk.Revit.DB;
using Autodesk.Revit.DB.Architecture;
using Autodesk.Revit.UI;
using Newtonsoft.Json;

namespace RoomGraph;

[Transaction(TransactionMode.Manual)]
public class ExportRoomRelationCommand : IExternalCommand
{
    public Result Execute(ExternalCommandData commandData, ref string message, ElementSet elements)
    {
        try
        {
            var doc = commandData.Application.ActiveUIDocument.Document;
            TestExportRoomRelationship(doc);
        }
        catch (Exception e)
        {
            MessageBox.Show(e.ToString());
        }

        return Result.Succeeded;
    }

    public void TestExportRoomRelationship(Autodesk.Revit.DB.Document doc)
    {
        List<InitRoom> initRooms = new List<InitRoom>();
        FilteredElementCollector collector = new FilteredElementCollector(doc);
        IList<Element> elements = collector.OfClass(typeof(SpatialElement)).ToElements();
        foreach (SpatialElement element in elements)
        {
            Room room = element as Room;
            if (room != null)
            {
                InitRoom initRoom = new InitRoom();
                initRoom.Room = element as Autodesk.Revit.DB.Architecture.Room;
                initRoom.childs = new List<Autodesk.Revit.DB.ElementId>();
                initRoom.relatedWalls = new List<RelatedWall>();
                Autodesk.Revit.DB.SpatialElementBoundaryOptions ops = new SpatialElementBoundaryOptions();
                SpatialElementBoundaryLocation boundloc = AreaVolumeSettings.GetAreaVolumeSettings(doc)
                    .GetSpatialElementBoundaryLocation(SpatialElementType.Room);
                ops.SpatialElementBoundaryLocation = boundloc;
                IList<IList<BoundarySegment>>? boundarySegments = room.GetBoundarySegments(ops);
                if (boundarySegments == null) continue;
                foreach (var boundarySegment in boundarySegments)
                {
                    foreach (var segment in boundarySegment)
                    {
                        ElementId elementId = segment.ElementId;
                        var ele = doc.GetElement(elementId);
                        bool isWalls = doc.GetElement(elementId)?.Category?.Name == "Walls";
                        if (isWalls)
                        {
                            BoundingBoxXYZ boundingBoxXyz = ele.get_BoundingBox(null);
                            // 4 points of bounding box
                            XYZ p1 = boundingBoxXyz.Min;
                            XYZ p2 = new XYZ(boundingBoxXyz.Max.X, boundingBoxXyz.Min.Y, boundingBoxXyz.Min.Z);
                            XYZ p3 = boundingBoxXyz.Max;
                            XYZ p4 = new XYZ(boundingBoxXyz.Min.X, boundingBoxXyz.Max.Y, boundingBoxXyz.Min.Z);
                            RelatedWall relatedWall = new RelatedWall()
                            {
                                id = ele.Id.IntegerValue,
                                name = ele.Name,
                                type = doc.GetElement(ele.GetTypeId()).Name,
                                bounds = new List<XYZ>()
                                {
                                    p1, p2, p3, p4
                                },
                                level = ele.get_Parameter(BuiltInParameter.WALL_BASE_CONSTRAINT).AsValueString(),
                                length = ele.get_Parameter(BuiltInParameter.CURVE_ELEM_LENGTH).AsValueString(),
                                height = ele.get_Parameter(BuiltInParameter.WALL_USER_HEIGHT_PARAM).AsValueString(),
                                area = ele.get_Parameter(BuiltInParameter.HOST_AREA_COMPUTED).AsValueString(),
                                volume = ele.get_Parameter(BuiltInParameter.HOST_VOLUME_COMPUTED).AsValueString()
                            };
                            if (relatedWall != null)
                            {
                                var wall = ele as Wall;
                                // get dependent elements just doors
                                ElementFilter doorFilter = new ElementCategoryFilter(BuiltInCategory.OST_Doors);
                                IList<ElementId> doorIds = wall.GetDependentElements(doorFilter);
                                foreach (var doorId in doorIds)
                                {
                                    var door = doc.GetElement(doorId);
                                    if (door == null) continue;
                                    BoundingBoxXYZ box = door.get_BoundingBox(null);
                                    // 4 points of bounding box
                                    XYZ p1Door = box.Min;
                                    XYZ p2Door = new XYZ(box.Max.X, box.Min.Y, box.Min.Z);
                                    XYZ p3Door = box.Max;
                                    XYZ p4Door = new XYZ(box.Min.X, box.Max.Y, box.Min.Z);
                                    RelatedDoor relatedDoor = new RelatedDoor()
                                    {
                                        id = door?.Id?.IntegerValue ?? 0,
                                        name = door.Name,
                                        bounds = new List<XYZ>()
                                        {
                                            p1Door, p2Door, p3Door, p4Door
                                        },
                                        level = door.get_Parameter(BuiltInParameter.FAMILY_LEVEL_PARAM)?.AsValueString(),
                                        type = door.get_Parameter(BuiltInParameter.DOOR_NUMBER)?.AsValueString()
                                    };
                                    relatedWall.relatedDoors.Add(relatedDoor);
                                }
                            }
                            // fix existing related wall
                            var flag = initRoom.relatedWalls.Any(x => x.id == relatedWall.id);
                            if (flag)
                            {
                                var index = initRoom.relatedWalls.FindIndex(x => x.id == relatedWall.id);
                                initRoom.relatedWalls[index] = relatedWall;
                            }
                            else
                            {
                                initRoom.relatedWalls.Add(relatedWall);
                            }
                        }

                        initRoom.childs.Add(elementId);
                    }
                }

                initRooms.Add(initRoom);
            }
        }

        List<RoomRelationship> roomRelationships = new List<RoomRelationship>();
        for (int i = 0; i < initRooms.Count; i++)
        {
            for (int j = 0; j < initRooms.Count; j++)
            {
                if (i != j)
                {
                    if (initRooms[i].childs.Count == 0 || initRooms[j].childs.Count == 0) continue;
                    if (CheckIfAnyExists(initRooms[i].childs, initRooms[j].childs))
                    {
                        RoomRelationship roomRelationship = new RoomRelationship();
                        roomRelationship.id = initRooms[i].Room.Id.IntegerValue;
                        roomRelationship.name = initRooms[i]?.Room?.Name;
                        roomRelationship.level = initRooms[i]?.Room?.Level.Name;
                        roomRelationship.number = initRooms[i]?.Room?.Number;
                        roomRelationship.area = initRooms[i].Room.get_Parameter(BuiltInParameter.ROOM_AREA).AsValueString();
                        roomRelationship.perimeter =
                            initRooms[i].Room.get_Parameter(BuiltInParameter.ROOM_PERIMETER).AsValueString();
                        roomRelationship.volume = initRooms[i].Room.get_Parameter(BuiltInParameter.ROOM_VOLUME).AsValueString();
                        roomRelationship.height =
                            initRooms[i].Room.get_Parameter(BuiltInParameter.ROOM_HEIGHT).AsValueString();
                        roomRelationship.bounds = GetBoundaries(initRooms[i].Room);
                        roomRelationship.relatedWalls = initRooms[i].relatedWalls;
                        Dictionary<int, string> relatedRoom = new Dictionary<int, string>()
                            { { initRooms[j].Room.Id.IntegerValue, initRooms[j].Room.Name } };
                        roomRelationship.relatedRooms = relatedRoom;
                        var flag = roomRelationships.Any(x => x.id == roomRelationship.id);
                        if (flag)
                        {
                            var index = roomRelationships.FindIndex(x => x.id == roomRelationship.id);
                            roomRelationships[index].relatedRooms
                                .Add(initRooms[j].Room.Id.IntegerValue, initRooms[j].Room.Name);
                        }
                        else
                        {
                            roomRelationships.Add(roomRelationship);
                        }
                    }
                }
            }
        }

        // export to json
        string tempFolder = Path.GetTempPath();
        string path = Path.Combine(tempFolder, "room_relationship.json");
        // export include room and related room
        File.WriteAllText(path, JsonConvert.SerializeObject(roomRelationships, Formatting.Indented,
            new JsonSerializerSettings()
            {
                MaxDepth = 1,
                Formatting = Formatting.Indented,
            }));
        MessageBox.Show(@"Done");
        Process.Start(path);
    }

    public List<XYZ> GetBoundaries(Room room)
    {
        SpatialElementBoundaryOptions options = new SpatialElementBoundaryOptions();
        options.SpatialElementBoundaryLocation = SpatialElementBoundaryLocation.Finish;
        IList<IList<Autodesk.Revit.DB.BoundarySegment>> boundaries = room.GetBoundarySegments(options);
        List<XYZ> bound = new List<XYZ>();
        foreach (var boundary in boundaries)
        {
            foreach (var segment in boundary)
            {
                bound.Add(segment.GetCurve().GetEndPoint(0));
            }
        }
        return bound;
    }

    public void ExportRoomRelationship(Autodesk.Revit.DB.Document doc)
    {
        // get all room
        var rooms = new Autodesk.Revit.DB.FilteredElementCollector(doc)
            .OfCategory(BuiltInCategory.OST_Rooms)
            .WhereElementIsNotElementType()
            .ToElements();
    }

    public static bool CheckIfAnyExists(List<Autodesk.Revit.DB.ElementId> listA,
        List<Autodesk.Revit.DB.ElementId> listB)
    {
        HashSet<int> hashSet = new HashSet<int>(listA.Select(e => e.IntegerValue));
        return listB.Any(e => hashSet.Contains(e.IntegerValue));
    }
}

[Serializable]
public class RoomRelationship : RoomBaseData
{
    public Dictionary<int, string> relatedRooms { get; set; } = new Dictionary<int, string>();

    public List<RelatedWall> relatedWalls { get; set; } = new List<RelatedWall>();
}

public class RoomBaseData
{
    public int? id { get; set; }
    public string? name { get; set; }
    public string? number { get; set; }
    public string? area { get; set; }
    public string? perimeter { get; set; }
    public string? volume { get; set; }
    public string? height { get; set; }
    public string? level { get; set; }
    public List<XYZ> bounds { get; set; }
}

public class RelatedWall
{
    public int id { get; set; }
    public string name { get; set; }
    public string type { get; set; }
    public List<XYZ> bounds { get; set; }
    public string level { get; set; }
    public string length { get; set; }
    public string height { get; set; }
    public string area { get; set; }
    public string volume { get; set; }
    public List<RelatedDoor> relatedDoors { get; set; } = new List<RelatedDoor>();
}

public class RelatedDoor
{
    public int id { get; set; }
    public string name { get; set; }
    public string? level { get; set; }
    public string? type { get; set; }
    public List<XYZ> bounds { get; set; }
}

public class bbox
{
    public XYZ? Min { get; set; }
    public XYZ? Max { get; set; }
}

public class InitRoom
{
    public Room? Room { get; set; }

    public List<RelatedWall> relatedWalls { get; set; } = new List<RelatedWall>();
    public List<Autodesk.Revit.DB.ElementId> childs { get; set; }
}