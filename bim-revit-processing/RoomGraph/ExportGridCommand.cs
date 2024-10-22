using System.Diagnostics;
using System.IO;
using System.Windows;
using Autodesk.Revit.Attributes;
using Autodesk.Revit.DB;
using Autodesk.Revit.UI;
using Newtonsoft.Json;

namespace RoomGraph;

[Transaction(TransactionMode.Manual)]
public class ExportGridCommand : IExternalCommand
{
    public Result Execute(ExternalCommandData commandData, ref string message, ElementSet elements)
    {
        ExportData(commandData);
        return Result.Succeeded;
    }

    public void ExportData(ExternalCommandData commandData)
    {
        // export into json
        var doc = commandData.Application.ActiveUIDocument.Document;
        var grids = GetGrids(doc);
        var gridItems = new List<GirdLineItem>();
        foreach (var grid in grids)
        {
            var startPoint = grid.Curve.GetEndPoint(0);
            var endPoint = grid.Curve.GetEndPoint(1);
            gridItems.Add(new GirdLineItem()
            {
                name = grid.Name,
                startPoint = startPoint,
                endPoint = endPoint
            });
        }
        string tempFolder = Path.GetTempPath();
        string path = Path.Combine(tempFolder, "grid_line.json");
        // export json list object
        File.WriteAllText(path, JsonConvert.SerializeObject(gridItems, Formatting.Indented,
            new JsonSerializerSettings()
            {
                MaxDepth = 1,
                Formatting = Formatting.Indented,
            }));
        MessageBox.Show("Done");
        Process.Start(path);
    }
    public List<Autodesk.Revit.DB.Grid> GetGrids(Document doc)
    {
        var grids = new FilteredElementCollector(doc)
            .OfClass(typeof(Autodesk.Revit.DB.Grid))
            .Cast<Autodesk.Revit.DB.Grid>()
            .ToList();
        return grids;
    }
}

public class GirdLineItem
{
     public string name { get; set; }
     public XYZ startPoint { get; set; }
     public XYZ endPoint { get; set; }
}