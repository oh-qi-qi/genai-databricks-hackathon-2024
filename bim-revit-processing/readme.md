## Revit Room & Door Data Extraction Overview

## Overview
This code extracts room and door information from a Revit model, focusing on the spatial relationships between doors and their connected rooms. It captures essential geometric and metadata properties, outputting them in a structured JSON format for further processing or analysis.

## Key Features
- Extracts door properties including location, size, and type
- Captures room metadata such as area, volume, and boundaries
- Maintains spatial relationships between doors and rooms
- Preserves 3D coordinate information
- Outputs standardized JSON format

## Sample Output

```json
{
  "door": {
    "id": 430758,
    "name": "72\" x 84\"",
    "level": "01 - Entry Level",
    "type": "322D",
    "bounds": [
      {
        "X": 324.49885461445359,
        "Y": -49.183028468366324,
        "Z": -2.0791610813758307E-16
      },
      // ... additional boundary points
    ]
  },
  "fromRoom": {
    "id": 526445,
    "name": "ELECTRICAL 01-26",
    "number": "01-26",
    "area": "173 m²",
    "perimeter": "52627",
    "volume": "0.00 CF",
    "height": "3658",
    "level": "01 - Entry Level",
    "bounds": [
      // Room boundary points
    ]
  },
  "toRoom": {
    "id": 526452,
    "name": "HALLWAY 01-33",
    // ... additional room properties
  }
}
```

## Data Structure
### Door Properties
- `id`: Unique identifier
- `name`: Size/dimensions
- `level`: Building level
- `type`: Door classification
- `bounds`: 3D boundary points

### Room Properties
- `id`: Unique identifier
- `name`: Room name with number
- `number`: Room number
- `area`: Floor area
- `perimeter`: Room perimeter
- `volume`: Room volume
- `height`: Room height
- `level`: Building level
- `bounds`: Room boundary points

## Model
The data here are curated for this Hackathon only:
- model: https://autode.sk/40aNFij
- grid_line.json - [grid_line.json](./data/grid_line.json)
- related_doors.json - [related_doors.json](./data/related_doors.json)
- room.json - [room.json](./data/room_relationship.json)

## How to run ?
- Clone the project
- Open Revit Software and demo model
- Build the project and use [Revit Add-in Manager](https://github.com/chuongmep/RevitAddInManager) to run command :
  - [ExportGridCommand](../RoomGraph/RoomGraph/ExportGridCommand.cs) : Export Grid Line to JSON file
  - [ExportRoomRelationCommand](../RoomGraph/RoomGraph/ExportRoomRelationCommand.cs) : Export Room Relationship to JSON file
  - [ExportDoorRelateCommand](../RoomGraph/RoomGraph/ExportDoorRelateCommand.cs): Export Door Relation to JSON file

## Q&A

- Why C# not Python ? 
C# is often preferred over Python for Revit development because it offers more power and flexibility. Its tighter integration with Revit's API and better performance in handling complex operations makes it a more efficient choice for large-scale or performance-sensitive projects.

- Why we need to export Door Relation ?
The door relation is important for the room relationship. The door is the main entrance to the room and the room relationship is based on the door relationship.

## Limitation

- The project demo just tested with Revit 2022.2, the project may be need to update for the latest version of Revit.

## Resources 

- [Revit API](https://www.revitapidocs.com/)
- [Revit Add-in Manager](https://github.com/chuongmep/RevitAddInManager)
- [https://hub.stackup.dev/](https://hub.stackup.dev/)
