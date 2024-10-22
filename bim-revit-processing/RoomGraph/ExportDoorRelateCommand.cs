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
public class ExportDoorRelateCommand : IExternalCommand
{
    public Result Execute(ExternalCommandData commandData, ref string message, ElementSet elements)
    {
        GenerateRelateDoors(commandData.Application.ActiveUIDocument.Document);
        return Result.Succeeded;
    }
    public void GenerateRelateDoors(Autodesk.Revit.DB.Document doc)
    {
        FilteredElementCollector collector = new FilteredElementCollector(doc);
        collector.OfCategory(BuiltInCategory.OST_Doors);
        List<Autodesk.Revit.DB.Element> doors = collector.ToElements().ToList();
        List<DoorItem> relatedDoors = new List<DoorItem>();
        foreach (Autodesk.Revit.DB.Element door in doors)
        {
            if (door is FamilyInstance)
            {
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
                FamilyInstance? familyInstance = door as FamilyInstance;
                var fromRoom = familyInstance?.FromRoom;
                var toRoom = familyInstance?.ToRoom;
                RoomBaseData fromRoomBase = default;
                if (fromRoom == null)
                {
                    fromRoomBase = new RoomBaseData()
                    {
                        id = -1,
                        name = "OUTSIDE",
                        number = "OUTSIDE",
                    };
                }
                else
                {
                    fromRoomBase = new RoomBaseData();
                    fromRoomBase.id = fromRoom.Id.IntegerValue;
                    fromRoomBase.name = fromRoom.Name;
                    fromRoomBase.level = fromRoom.Level.Name;
                    fromRoomBase.number = fromRoom.Number;
                    fromRoomBase.area = fromRoom.get_Parameter(BuiltInParameter.ROOM_AREA).AsValueString();
                    fromRoomBase.perimeter =
                        fromRoom.get_Parameter(BuiltInParameter.ROOM_PERIMETER).AsValueString();
                    fromRoomBase.volume = fromRoom.get_Parameter(BuiltInParameter.ROOM_VOLUME).AsValueString();
                    fromRoomBase.height =
                        fromRoom.get_Parameter(BuiltInParameter.ROOM_HEIGHT).AsValueString();
                    fromRoomBase.bounds = GetBoundaries(fromRoom);
                }
                RoomBaseData toRoomBase = new RoomBaseData();
                if (toRoom == null)
                {
                    toRoomBase = new RoomBaseData()
                    {
                        id = -1,
                        name = "OUTSIDE",
                        number = "OUTSIDE",
                    };
                }
                else
                {
                    toRoomBase.id = toRoom?.Id?.IntegerValue;
                    toRoomBase.name = toRoom?.Name;
                    toRoomBase.level = toRoom?.Level?.Name;
                    toRoomBase.number = toRoom?.Number;
                    toRoomBase.area = toRoom?.get_Parameter(BuiltInParameter.ROOM_AREA).AsValueString();
                    toRoomBase.perimeter =
                        toRoom?.get_Parameter(BuiltInParameter.ROOM_PERIMETER).AsValueString();
                    toRoomBase.volume = toRoom?.get_Parameter(BuiltInParameter.ROOM_VOLUME).AsValueString();
                    toRoomBase.height =
                        toRoom?.get_Parameter(BuiltInParameter.ROOM_HEIGHT).AsValueString();
                    toRoomBase.bounds = GetBoundaries(toRoom);
                }
                relatedDoors.Add(new DoorItem()
                {
                    door = relatedDoor,
                    fromRoom = fromRoomBase,
                    toRoom = toRoomBase
                });

            }
        }

        string tempFolder = Path.GetTempPath();
        string path = Path.Combine(tempFolder, "related_doors.json");
        // export include room and related room
        File.WriteAllText(path, JsonConvert.SerializeObject(relatedDoors, Formatting.Indented,
            new JsonSerializerSettings()
            {
                MaxDepth = 1,
                Formatting = Formatting.Indented,
            }));
        MessageBox.Show(@"Done");
        Process.Start(path);

    }
    public List<XYZ> GetBoundaries(Room? room)
    {
        if (room == null)
        {
            return new List<XYZ>();
        }
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

    public class DoorItem
    {
        public RelatedDoor door { get; set; }
        public RoomBaseData fromRoom { get; set; }
        public RoomBaseData toRoom { get; set; }
    }


}
