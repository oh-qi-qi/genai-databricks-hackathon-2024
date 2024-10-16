# Databricks notebook source
# MAGIC %md
# MAGIC #### Dependency and Installation

# COMMAND ----------

# MAGIC %run ./dependency_installation_setup

# COMMAND ----------

import os

catalog_name = spark.sql("SELECT current_catalog()").collect()[0][0]
schema_name = spark.catalog.currentDatabase()
working_directory = os.getcwd()
dataset_location = f"/Volumes/{catalog_name}/{schema_name}/regubim-ai-volume/"

print(f"Catalog Name: {catalog_name}")
print(f"Schema Name: {schema_name}")
print(f"Working Directory: {working_directory}")
print(f"Dataset Location: {dataset_location}")

files = dbutils.fs.ls(dataset_location)

# Print the files and folders in the volume
for file in files:
    print(file.name)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Processing Revit data portion

# COMMAND ----------

# read the revit converted to json data file and create a spark dataframe
import pandas as pd

revit_room_filename = "revit_model_room_data.json"
revit_room_file_path = f"{dataset_location}{revit_room_filename}"

# Load the JSON file into a Pandas DataFrame
df_pandas_revit_room_data = pd.read_json(revit_room_file_path)

# Convert the Pandas DataFrame to a Spark DataFrame
df_spark_revit_room_data = spark.createDataFrame(df_pandas_revit_room_data)

display(df_spark_revit_room_data)

# COMMAND ----------

# From the Spark Dataframe, process it and create a room vertices and edges which is the room connections 

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, ArrayType, DoubleType

# Step 1: Extract the door information with levels (ARRAY<STRUCT> bounds)
door_df = df_spark_revit_room_data.select(
    F.col("door.id").alias("id"),  # Ensure it's named door_id
    F.col("door.name").alias("name"), 
    F.lit(None).cast("string").alias("number"),  # Add a placeholder 'number' column
    F.lit(None).alias("area"),  # Add a placeholder 'number' column
    F.col("door.level").alias("level"),
    F.col("door.bounds").alias("bounds"),  # Keep the ARRAY<STRUCT> format
    F.lit("Door").alias("type"),
    
).distinct()

# From Room
fromRoom_df = df_spark_revit_room_data.select(
    F.col("fromRoom.id").alias("id"), 
    F.col("fromRoom.name").alias("name"), 
    F.col("fromRoom.number").alias("number"),  # Room-specific 'number'
    F.col("fromRoom.area").alias("area"),
    F.col("fromRoom.level").alias("level"),  
    F.col("fromRoom.bounds").alias("bounds"),
    F.lit("Room").alias("type")  # Adding type column for room
).distinct()

# To Room 
toRoom_df = df_spark_revit_room_data.select(
    F.col("toRoom.id").alias("id"), 
    F.col("toRoom.name").alias("name"), 
    F.col("toRoom.number").alias("number"),  # Room-specific 'number'
    F.col("toRoom.area").alias("area"),
    F.col("toRoom.level").alias("level"),  
    F.col("toRoom.bounds").alias("bounds"),
    F.lit("Room").alias("type")  # Adding type column for room
).distinct()

# Step 3: Union all DataFrames to combine rooms and doors
vertices_df = fromRoom_df.union(toRoom_df).union(door_df).distinct()

# Step 4: Create Edges (Connections between Rooms via Doors)
edges_df = df_spark_revit_room_data.select(
    F.col("fromRoom.id").alias("src"),
    F.col("fromRoom.name").alias("src_name"),
    F.col("fromRoom.number").alias("src_number"),
    F.col("fromRoom.level").alias("src_level"),
    F.col("fromRoom.area").alias("src_area"),
    F.col("fromRoom.bounds").alias("src_bounds"),
    F.col("toRoom.id").alias("dst"),
    F.col("toRoom.name").alias("dst_name"),
    F.col("toRoom.number").alias("dst_number"),
    F.col("toRoom.level").alias("dst_level"),
    F.col("toRoom.area").alias("dst_area"),
    F.col("toRoom.bounds").alias("dst_bounds"),
    F.lit("connects").alias("relationship"),
    F.col("door.id").alias("door_id")
).join(door_df, F.col("door_id") == F.col("id"), "left").select(
    F.col("src"), 
    F.col("src_name"), 
    F.col("src_number"), 
    F.col("src_level"), 
    F.col("src_area"),
    F.col("src_bounds"), 
    F.col("dst"), 
    F.col("dst_name"), 
    F.col("dst_number"), 
    F.col("dst_level"),
    F.col("dst_area"),  
    F.col("dst_bounds"),
    F.col("relationship"),
    F.col("id").alias("door_id"),
    F.col("name").alias("door_name"),
    F.col("level").alias("door_level"),
    F.col("bounds").alias("door_bounds")
).orderBy(F.col("door_level"))  # Order by door_level (ascending by default)

# Step 5: Filter vertices_df to include only rooms (where type == "Room")
rooms_vertices_df = vertices_df.filter(F.col("type") == "Room")

# Display the vertices and edges DataFrames
display(vertices_df)
display(rooms_vertices_df)
display(edges_df)


# COMMAND ----------

# Define the table name
revit_room_vertices_table_name = f"{catalog_name}.{schema_name}.revit_room_vertices"
revit_room_edges_table_name = f"{catalog_name}.{schema_name}.revit_room_edges"

# COMMAND ----------

# Create a Delta table for the vertices and edges and read from it and create a room_graph_json
rooms_vertices_df.write.mode("append").saveAsTable(revit_room_vertices_table_name)
edges_df.write.mode("append").saveAsTable(revit_room_edges_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Explore room relationship

# COMMAND ----------


import json
import numpy as np

# Function to handle non-serializable data types
def convert_value(value):
    if isinstance(value, np.ndarray):  # Convert ndarray to list
        return value.tolist()
    if isinstance(value, np.generic):  # Convert numpy types to native Python types
        return value.item()
    return value  # Return the value as is if it's already serializable


# Read the Delta tables for vertices and edges
room_vertices_df_from_spark = spark.read.table(revit_room_vertices_table_name)
room_edges_df_from_spark  = spark.read.table(revit_room_edges_table_name)

# Step 5: Prepare JSON data for D3.js visualization
nodes = [
    {
        "id": convert_value(row['id']),
        "name": convert_value(row['name']),
        "level": convert_value(row['level']),
        "area": convert_value(row['area']),
        "bounds": convert_value(row['bounds']),
        "type": convert_value(row['type'])
    }
    for index, row in room_vertices_df_from_spark.toPandas().iterrows()
]

links = [
    {
        "source": convert_value(row['src']),
        "target": convert_value(row['dst']),
        "door_id": convert_value(row['door_id']),
        "door_name": convert_value(row['door_name']),
        "door_level": convert_value(row['door_level']),
        "door_bounds": convert_value(row['door_bounds'])
    }
    for index, row in room_edges_df_from_spark.toPandas().iterrows()
]

room_graph_data = {
    "nodes": nodes,
    "links": links
}


# Convert the graph data to JSON
room_graph_data_json = json.dumps(room_graph_data, indent=4)

# COMMAND ----------

# Create the HTML template with the dynamic `room_graph_json`
html_room_relationship_graph_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1.0" />
    <!-- Google Icons & Font -->
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet" />
    <link href="https://fonts.googleapis.com/css?family=Roboto:400,700&subset=latin,cyrillic-ext" rel="stylesheet" type="text/css" />
    <!-- Bootstrap and Font Awesome -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.6.0/css/all.min.css" rel="stylesheet" />
    <!-- External JavaScript and D3.js for graph rendering -->
    <script src="https://code.jquery.com/jquery-2.1.1.min.js"></script>
    <script src="https://d3js.org/d3.v6.min.js"></script>
    <script src="https://unpkg.com/d3-v6-tip@1.0.6/build/d3-v6-tip.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3-legend/2.13.0/d3-legend.js"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/bumbeishvili/d3-tip-for-v6@4/d3-tip.min.css" />
    <title>Data Visulisation</title>
    <style>
        .graph-container {
            margin: auto;
            width: 90%;
            padding: 10px;
        }
        div#data_vis_display {
            overflow: auto;
        }
        .d3-tip {
            font-family: Arial, Helvetica, sans-serif;
            line-height: 1.4;
            padding: 20px;
            pointer-events: none !important;
            color: #203d5d;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            background-color: #fff;
            border-radius: 4px;
        }
        svg {
            border: 1px solid black;
        }
        /* Style for the legend */
        .legend {
            font-size: 14px;
            font-family: Arial, sans-serif;
        }
        #selectButton{
            margin-bottom:10px;
        }
    </style>
</head>
<body>
    <div class="graph-container">
        <div class="input-field col l2 m3 s12">
            <!-- Initialize a select button -->
            <select id="selectButton"></select>
        </div>
        <!-- Div to hold the D3.js graph -->
        <div id="data_vis_display"></div>
        <div id="tooltip"></div>
    </div>
    <script>
        $(document).ready(function() {
            // graph data
            var graph = """ + room_graph_data_json + """;
            const margin = {
                top: 0,
                right: 200,
                bottom: 0,
                left: 0
            };
            const width = 1400,
                height = 900,
                circle_radius = 40;
            const svg = d3.select("#data_vis_display").append("svg").attr("viewBox", `0 0 ${width} ${height}`);
            const buildingLevels = Array.from(new Set(graph.nodes.map((d) => d.level).sort((a, b) => parseFloat(a) - parseFloat(b))));
            const levelOptions = ["Show All", ...buildingLevels];
            // Populate dropdown
            d3.select("#selectButton")
                .selectAll("option")
                .data(levelOptions)
                .enter()
                .append("option")
                .text((d) => d)
                .attr("value", (d) => d);
            // $("select").formSelect(); // Initialize select element
            d3.select("#selectButton").on("change", function() {
                update(this.value);
            });
            const colorScaleLevel = d3.scaleOrdinal().domain(buildingLevels).range(d3.schemeCategory10);
            // Legend
            svg.append("g")
                .attr("class", "legend")
                .attr("transform", `translate(${width - margin.right},20)`)
                .call(
                    d3
                    .legendColor()
                    .labels(buildingLevels.map((l, i) => `${l}`))
                    .title("Level")
                    .scale(colorScaleLevel)
                );
            // Arrow marker definition
            svg.append("defs")
                .append("marker")
                .attr("id", "end")
                .attr("viewBox", "0 -5 10 10")
                .attr("refX", 10)
                .attr("markerWidth", 6)
                .attr("markerHeight", 6)
                .attr("orient", "auto")
                .append("path")
                .attr("d", "M0,-5L10,0L0,5")
                .style("fill", "#666")
                .style("stroke", "none");
            // Tooltip setup
            const tip = d3
                .tip()
                .attr("class", "d3-tip")
                .html(function(event, d) {
                    return (
                        "<strong>Name:</strong> <span>" +
                        d.name +
                        "</span>" +
                        "<br><strong>Type:</strong> <span>" +
                        d.type +
                        "</span>" +
                        "<br><strong>Area:</strong> <span>" +
                        d.area +
                        "</span>"
                    );
                });
            svg.call(tip);
            // Force simulation setup
            const simulation = d3
                .forceSimulation()
                .force("x", d3.forceX(width / 2).strength(0.15))
                .force("y", d3.forceY(height / 2).strength(0.3))
                .force(
                    "link",
                    d3
                    .forceLink()
                    .id((d) => d.id)
                    .distance(100)
                )
                .force("charge", d3.forceManyBody().strength(-200))
                .force("collide", d3.forceCollide(circle_radius + 20).strength(0.8))
                .on("tick", ticked);
            const nodeGroup = svg.append("g").attr("class", "nodes");
            const linkGroup = svg.append("g").attr("class", "links");
            // Initialize the graph
            update(levelOptions[0]);
            // Update function
            function update(selectedLevel) {
                const filteredNodes =
                    selectedLevel === "Show All" ?
                    graph.nodes :
                    graph.nodes.filter((d) => d.level === selectedLevel);
                const nodeIds = new Set(filteredNodes.map((d) => d.id));
                const filteredLinks = graph.links.filter(
                    (d) => nodeIds.has(d.source.id || d.source) && nodeIds.has(d.target.id || d.target)
                );
                // Links update
                const link = linkGroup
                    .selectAll("path")
                    .data(filteredLinks, (d) => d.source.id + "-" + d.target.id);
                link.exit().remove();
                link.enter()
                    .append("path")
                    .attr("class", "links")
                    .attr("stroke-width", 1)
                    .attr("stroke", "#666")
                    .attr("fill", "none") // Ensures no fill for the path
                    .attr("marker-end", "url(#end)")
                    .merge(link);
                // Nodes update
                const node = nodeGroup.selectAll("g").data(filteredNodes, (d) => d.id);
                node.exit().remove();
                const nodeEnter = node
                    .enter()
                    .append("g")
                    .call(d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended));
                nodeEnter
                    .append("circle")
                    .attr("r", circle_radius)
                    .style("fill", (d) => colorScaleLevel(d.level));
                // Text labels with icons and text wrapping
                nodeEnter
                    .append("text")
                    .attr("text-anchor", "middle")
                    .attr("dominant-baseline", "central")
                    .attr("font-family", "FontAwesome")
                    .style("fill", "white")
                    .each(function(d) {
                        const textElement = d3.select(this);
                        const icon = d.type === "Room" ? getIconFromName(d.name) : "";
                        // Add icon above the text
                        textElement
                            .append("tspan")
                            .attr("font-size", "12px") // Larger font for the icon
                            .attr("dy", "-1.5em") // Move icon upwards
                            .text(icon);
                        // Add text with wrapping logic
                        const words = d.name.split(/\s+/);
                        let line = [],
                            lineHeight = 1.2,
                            tspan = textElement
                            .append("tspan")
                            .attr("x", 0)
                            .attr("dy", "2.1em")
                            .attr("font-size", "8px");
                        words.forEach((word) => {
                            line.push(word);
                            tspan.text(line.join(" "));
                            if (tspan.node().getComputedTextLength() > 2 * (circle_radius - 8)) {
                                line.pop();
                                tspan.text(line.join(" "));
                                line = [word];
                                tspan = textElement
                                    .append("tspan")
                                    .attr("x", 0)
                                    .attr("dy", lineHeight + "em")
                                    .attr("font-size", "8px")
                                    .text(word);
                            }
                        }, this);
                    });
                nodeEnter.on("mouseover", tip.show).on("mouseout", tip.hide);
                node.merge(nodeEnter);
                simulation.nodes(filteredNodes);
                simulation.force("link").links(filteredLinks);
                simulation.alpha(1).restart();
            }
            // Ticking function
            function ticked() {
                nodeGroup.selectAll("g").attr("transform", (d) => `translate(${d.x},${d.y})`);
                linkGroup.selectAll("path").attr("d", linkPath);
            }
            // Link path generation
            function linkPath(d, i) {
                const dx = d.target.x - d.source.x;
                const dy = d.target.y - d.source.y;
                const dr = Math.sqrt(dx * dx + dy * dy);
                const curvatureOffset = (i % 10) * 10;
                const scaleSource = circle_radius / dr;
                const scaleTarget = circle_radius / dr;
                const [sourceX, sourceY] = [d.source.x + dx * scaleSource, d.source.y + dy * scaleSource];
                const [targetX, targetY] = [d.target.x - dx * scaleTarget, d.target.y - dy * scaleTarget];
                return dr > circle_radius * 2 ?
                    `M${sourceX},${sourceY}A${dr + curvatureOffset},${dr + curvatureOffset} 0 0,1 ${targetX},${targetY}` :
                    `M${d.source.x},${d.source.y}L${d.target.x},${d.target.y}`;
            }
            // Drag functions
            function dragstarted(event, d) {
                if (!event.active) simulation.alphaTarget(0.2).restart();
                d.fx = d.x;
                d.fy = d.y;
            }
            function dragged(event, d) {
                d.fx = event.x;
                d.fy = event.y;
            }
            function dragended(event, d) {
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }
            function getIconFromName(name) {
                const iconMap = {
                    Staircase: "\ue289",
                    Computer: "\ue4e5",
                    Electrical: "\uf0eb",
                    Washroom: "\uf7d8",
                    Meeting: "\ue537",
                    Training: "\uf51c",
                    Hallway: "\uf557",
                    Smoking: "\uf48d",
                    Security: "\ue54a",
                    Prayer: "\uf683",
                    Mechanical: "\uf0ad",
                    Cafeteria: "\uf0f4",
                    Outside: "\uf850",
                    Loading: "\uf4de"
                };
                const trimmedName = name.trim().toLowerCase();
                // Find the keyword in iconMap that matches the name and return its value (icon)
                const matchingKeyword = Object.keys(iconMap).find((keyword) =>
                    trimmedName.includes(keyword.toLowerCase())
                );
                return iconMap[matchingKeyword] || "\uf0db"; // Return the corresponding icon or a default icon
            }
        });
    </script>
</body>
</html>
"""

# COMMAND ----------

# Render the HTML content directly inside Databricks
displayHTML(html_room_relationship_graph_template)

# COMMAND ----------

from fuzzywuzzy import process
import networkx as nx
import pandas as pd
import numpy as np
import json

def numpy_to_python(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    else:
        return obj
    
def calculate_center_point(points):
    try:
        if isinstance(points, np.ndarray):
            points = points.tolist()
        if not isinstance(points, list) or len(points) != 4:
            return {'X': 0, 'Y': 0, 'Z': 0}
        x_sum = sum(p['X'] for p in points)
        y_sum = sum(p['Y'] for p in points)
        z_sum = sum(p['Z'] for p in points)
        return {'X': x_sum / 4, 'Y': y_sum / 4, 'Z': z_sum / 4}
    except Exception as e:
        print(f"Error calculating center point: {e}")
        print(f"Points data: {points}")
        return {'X': 0, 'Y': 0, 'Z': 0}

def manhattan_distance(point1, point2):
    return (abs(point1['X'] - point2['X']) + 
            abs(point1['Y'] - point2['Y']) + 
            abs(point1['Z'] - point2['Z']))

def find_matching_rooms(room_name, room_names, threshold=80):
    matches = process.extract(room_name, room_names, limit=None)
    return [match for match in matches if match[1] >= threshold]

def find_room_id(room_name, room_edges_df):
    matches = process.extract(room_name, room_edges_df['src_name'].unique(), limit=1)
    if matches and matches[0][1] >= 80:
        return room_edges_df[room_edges_df['src_name'] == matches[0][0]]['src'].iloc[0]
    return None

def custom_path_finder(G, source, target):
    def dfs_paths(current, path):
        if current == target:
            yield path
        for neighbor in G.neighbors(current):
            if neighbor not in path:
                if "OUTSIDE" not in neighbor or neighbor in (source, target):
                    yield from dfs_paths(neighbor, path + [neighbor])

    return list(dfs_paths(source, [source]))

def find_all_shortest_paths(room_edges_df_from_spark, source_room, target_room):
    # Convert Spark DataFrame to Pandas DataFrame
    room_edges_df = room_edges_df_from_spark.toPandas()

    # Create a graph of rooms
    G = nx.Graph()

    # Add rooms as nodes and connections as edges
    for _, row in room_edges_df.iterrows():
        src_name, dst_name = row['src_name'], row['dst_name']
        door_center = calculate_center_point(row['door_bounds'])
        
        # Add nodes with all available information
        for name in [src_name, dst_name]:
            if not G.has_node(name):
                G.add_node(name, 
                           id=row['src'] if name == src_name else row['dst'],
                           level=row['src_level'] if name == src_name else row['dst_level'],
                           area=row['src_area'] if name == src_name else row['dst_area'],
                           bounds=numpy_to_python(row['src_bounds'] if name == src_name else row['dst_bounds']),
                           type="Room")

        # Add edge
        G.add_edge(src_name, dst_name, 
                   src_name=src_name, 
                   dst_name=dst_name,
                   src_id=row['src'],
                   dst_id=row['dst'],
                   door_id=row['door_id'], 
                   door_name=row['door_name'], 
                   door_level=row['door_level'], 
                   door_center=door_center,
                   door_bounds=numpy_to_python(row['door_bounds']))

    # Get unique room names
    room_names = list(G.nodes())

    # Find matching rooms for source and target
    source_matches = find_matching_rooms(source_room, room_names)
    target_matches = find_matching_rooms(target_room, room_names)

    if not source_matches:
        return "Source room not found"
    if not target_matches:
        return "Target room not found"

    all_paths = []

    for source_room_name, source_score in source_matches:
        for target_room_name, target_score in target_matches:
            try:
                # Find all simple paths between source and target
                simple_paths = custom_path_finder(G, source_room_name, target_room_name)
                
                for path in simple_paths:
                    total_distance = 0
                    door_path = []
                    fuzzy_path = []
                    
                    for i in range(len(path)):
                        room = path[i]
                        fuzzy_match = process.extractOne(room, room_names)
                        fuzzy_path.append(fuzzy_match[0])
                        
                        if i < len(path) - 1:
                            room1, room2 = path[i], path[i+1]
                            edge_data = G[room1][room2]
                            door_id = edge_data['door_id']
                            door_center = edge_data['door_center']
                            
                            if i > 0:  # Calculate distance from previous door to this door
                                distance = manhattan_distance(prev_door_center, door_center)
                                total_distance += distance
                            
                            door_path.append((door_id, edge_data['door_name'], edge_data['door_level']))
                            prev_door_center = door_center
                    
                    all_paths.append((fuzzy_path, total_distance, source_room_name, target_room_name, door_path, source_score, target_score))
            
            except nx.NetworkXNoPath:
                continue
            except Exception as e:
                print(f"Error processing path: {e}")
                print(f"Path: {path}")
                print(f"Edge data: {G[path[0]][path[1]]}")

    if not all_paths:
        return f"No paths found between any matching source and target rooms"

    # Sort paths by distance
    all_paths.sort(key=lambda x: x[1])

    # Create a DataFrame from all_paths
    paths_df = pd.DataFrame(all_paths, columns=['Path', 'Distance', 'Source', 'Target', 'DoorPath', 'SourceScore', 'TargetScore'])

    # Create graph JSON
    graph_json = {"nodes": [], "links": []}
    unique_rooms = set()
    for path, _, _, _, door_path, _, _ in all_paths:
        for room in path:
            if room not in unique_rooms:
                unique_rooms.add(room)
                node_data = G.nodes[room]
                graph_json["nodes"].append({
                    "id": node_data['id'],
                    "name": room,
                    "level": node_data['level'],
                    "area": node_data['area'],
                    "bounds": node_data['bounds'],
                    "type": node_data['type']
                })

    for i, (path, total_distance, _, _, door_path, _, _) in enumerate(all_paths):
        for j in range(len(path) - 1):
            source, target = path[j], path[j+1]
            edge_data = G[source][target]

            # Determine the correct source and target IDs
            source_id = edge_data['src_id'] if edge_data['src_name'] == source else edge_data['dst_id']
            target_id = edge_data['dst_id'] if edge_data['dst_name'] == target else edge_data['src_id']

            graph_json["links"].append({
                "source": source_id,
                "target": target_id,
                "source_name": source,
                "target_name": target,
                "door_id": edge_data['door_id'],
                "door_name": edge_data['door_name'],
                "door_level": edge_data['door_level'],
                "door_bounds": edge_data['door_bounds'],
                "route": i + 1,  # Add route number
                "route_distance": total_distance,  # Add total distance for the entire route
                "path": path  # Add the entire path as an array
            })

    return paths_df, json.dumps(numpy_to_python(graph_json))
# Example usage
source_room = "FCC"
target_room = "STAIRCASE"

paths_df, path_graph_json = find_all_shortest_paths(room_edges_df_from_spark, source_room, target_room)

# Flatten the DoorPath column into a string
paths_df['DoorPath'] = paths_df['DoorPath'].apply(lambda x: json.dumps(x) if isinstance(x, list) else str(x))
paths_spark_df = spark.createDataFrame(paths_df)
display(paths_spark_df)


# COMMAND ----------

html_room_path_graph_template = """
<!doctype html>
<html lang="en">
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1.0" />
        <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet" />
        <link
            href="https://fonts.googleapis.com/css?family=Roboto:400,700&subset=latin,cyrillic-ext"
            rel="stylesheet"
            type="text/css" />
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.6.0/css/all.min.css" rel="stylesheet" />
        <script src="https://code.jquery.com/jquery-2.1.1.min.js"></script>
        <script src="https://d3js.org/d3.v6.min.js"></script>
        <script src="https://unpkg.com/d3-v6-tip@1.0.6/build/d3-v6-tip.js"></script>
        <title>Room Route Graph Visualization</title>
        <style>
            .graph-container {
                margin: auto;
                width: 90%;
                padding: 10px;
            }
            div#data_vis_display {
                overflow: auto;
            }
            .d3-tip {
                line-height: 1.4;
                padding: 12px;
                background: rgba(0, 0, 0, 0.8);
                color: #fff;
                border-radius: 2px;
                pointer-events: none !important;
            }
            svg {
                border: 1px solid black;
            }
            .legend {
                font-size: 12px;
                font-family: sans-serif;
            }
            .route-label {
                font-size: 10px;
                font-weight: bold;
                fill: #fff;
                text-anchor: middle;
                dominant-baseline: central;
            }
        </style>
    </head>
    <body>
        <div class="graph-container">
            <div id="data_vis_display"></div>
        </div>

        <script>
            $(document).ready(function () {
                var graph = """ + path_graph_json + """;

                const width = 1000,
                    height = 600,
                    circle_radius = 40;
                const svg = d3.select("#data_vis_display").append("svg").attr("viewBox", `0 0 ${width} ${height}`);

                const nodeColorScale = d3
                    .scaleOrdinal()
                    .domain(["source", "destination", "intermediate"])
                    .range(["#1f77b4", "#2ca02c", "#ff7f0e"]);

                const tip = d3
                    .tip()
                    .attr("class", "d3-tip")
                    .offset([-10, 0])
                    .html(function (event, d) {
                      
                        if (d.route) {
                          console.log(d)
                            return `<strong>Route ${d.route}</strong><br>
                                Path: ${d.path.join(" → ")}<br>
                                Distance: ${d.route_distance.toFixed(2)}m`;
                        } else {
                            return `<strong>${d.name}</strong><br>
                                Type: ${d.type}<br>
                                Area: ${d.area}<br>
                                Level: ${d.level}`;
                        }
                    });

                svg.call(tip);

                // Arrow marker definition
                svg.append("defs")
                    .append("marker")
                    .attr("id", "end")
                    .attr("viewBox", "0 -5 10 10")
                    .attr("refX", 10)
                    .attr("markerWidth", 6)
                    .attr("markerHeight", 6)
                    .attr("orient", "auto")
                    .append("path")
                    .attr("d", "M0,-5L10,0L0,5")
                    .style("fill", "#666")
                    .style("stroke", "none");

                const simulation = d3
                    .forceSimulation()
                    .force(
                        "link",
                        d3
                            .forceLink()
                            .id((d) => d.id)
                            .distance(200)
                    )
                    .force("charge", d3.forceManyBody().strength(-500))
                    .force("center", d3.forceCenter(width / 2, height / 2))
                    .force("collide", d3.forceCollide().radius(circle_radius * 1.5));

                const linkGroup = svg.append("g").attr("class", "links");
                const nodeGroup = svg.append("g").attr("class", "nodes");

                function getNodeTypeFromLinks(nodeId, links) {
                    let sourceCount = 0;
                    let targetCount = 0;

                    // Iterate over all links to count occurrences of nodeId as source and target
                    links.forEach((link) => {
                        if (link.source === nodeId) {
                            sourceCount++;
                        }
                        if (link.target === nodeId) {
                            targetCount++;
                        }
                    });

                    console.log(nodeId);
                    // Determine the type based on the counts
                    if (targetCount > 1) {
                        return "intermediate";
                    } else if (sourceCount > 0) {
                        return "source";
                    } else if (targetCount > 0) {
                        return "destination";
                    }

                    return "unknown"; // Fallback if the nodeId is not found in the links
                }

                // Function to assign unique link numbers to overlapping links
                function computeLinkNumbers(links) {
                    let linkGroups = {};
                    links.forEach(function (d) {
                        let sourceId = typeof d.source === "object" ? d.source.id : d.source;
                        let targetId = typeof d.target === "object" ? d.target.id : d.target;
                        let key = [sourceId, targetId].sort().join(",");
                        if (!linkGroups[key]) {
                            linkGroups[key] = [];
                        }
                        linkGroups[key].push(d);
                    });
                    for (let key in linkGroups) {
                        let group = linkGroups[key];
                        group.forEach(function (link, i) {
                            link.linknum = i;
                            link.totalLinks = group.length;
                        });
                    }
                }

                // Call the function to assign link numbers
                computeLinkNumbers(graph.links);

                function update() {
                    const link = linkGroup.selectAll("g").data(graph.links).join("g");

                    // Append the path first
                    link.append("path")
                        .attr("fill", "none")
                        .attr("stroke", "#999")
                        .attr("stroke-width", 2)
                        .attr("marker-end", "url(#end)");

                    // Append labels
                    const labelGroup = link.append("g").attr("class", "label");

                    labelGroup
                        .append("rect")
                        .attr("width", 50)
                        .attr("height", 30)
                        .attr("rx", 10)
                        .attr("ry", 10)
                        .attr("fill", "#fff")
                        .attr("stroke", "#999");

                    labelGroup
                        .append("text")
                        .attr("class", "route-label")
                        .attr("dy", ".35em")
                        .style("font-size", "12px")
                        .style("font-weight", "bold")
                        .style("fill", "#000")
                        .attr("text-anchor", "middle")
                        .text((d) => `Route ${d.route}`)
                      .on("mouseover", tip.show)
                        .on("mouseout", tip.hide);

                    const node = nodeGroup
                        .selectAll("g")
                        .data(graph.nodes)
                        .join("g")
                        .call(d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended))
                        .on("mouseover", tip.show)
                        .on("mouseout", tip.hide);

                    node.append("circle")
                        .attr("r", circle_radius)
                        .attr("fill", (d) => nodeColorScale(getNodeTypeFromLinks(d.id, graph.links)));

                    node.append("text")
                        .attr("text-anchor", "middle")
                        .attr("dominant-baseline", "central")
                        .attr("font-family", "FontAwesome")
                        .style("fill", "white")
                        .each(function (d) {
                            const textElement = d3.select(this);
                            const icon = getIconFromName(d.name);
                            textElement.append("tspan").attr("font-size", "20px").attr("dy", "-0.5em").text(icon);

                            const words = d.name.split(/\s+/);
                            let lineHeight = 1.1;
                            words.forEach((word, i) => {
                                textElement
                                    .append("tspan")
                                    .attr("x", 0)
                                    .attr("dy", i ? `${lineHeight}em` : "1.5em")
                                    .attr("font-size", "10px")
                                    .text(word);
                            });
                        });

                    simulation.nodes(graph.nodes).on("tick", ticked);
                    simulation.force("link").links(graph.links);
                    simulation.alpha(1).restart();

                    // Legend
                    const legend = svg.append("g").attr("class", "legend").attr("transform", "translate(20,20)");

                    const legendData = [
                        { type: "Source", color: nodeColorScale("source") },
                        { type: "Destination", color: nodeColorScale("destination") },
                        { type: "Intermediate", color: nodeColorScale("intermediate") }
                    ];

                    const legendItems = legend
                        .selectAll(".legend-item")
                        .data(legendData)
                        .enter()
                        .append("g")
                        .attr("class", "legend-item")
                        .attr("transform", (d, i) => `translate(0,${i * 20})`);

                    legendItems
                        .append("rect")
                        .attr("width", 18)
                        .attr("height", 18)
                        .style("fill", (d) => d.color);

                    legendItems
                        .append("text")
                        .attr("x", 24)
                        .attr("y", 9)
                        .attr("dy", ".35em")
                        .text((d) => d.type)
                        .style("font-size", "12px");
                }
              
               function manhattanDistance(point1, point2) {
                    return Math.abs(point1.x - point2.x) + Math.abs(point1.y - point2.y);
                }

                function linkPath(d) {
                    const dx = d.target.x - d.source.x;
                    const dy = d.target.y - d.source.y;
                    const dr = Math.sqrt(dx * dx + dy * dy);

                    const angle = Math.atan2(dy, dx);

                    // Calculate offset angle for this link
                    const totalLinks = d.totalLinks;
                    const linknum = d.linknum;

                    const angleOffset = (linknum - (totalLinks - 1) / 2) * (Math.PI / 12);

                    // Calculate start and end points on the circles' circumferences
                    const sourceAngle = angle + angleOffset;
                    const targetAngle = angle + Math.PI + angleOffset;

                    const sourceX = d.source.x + Math.cos(sourceAngle) * circle_radius;
                    const sourceY = d.source.y + Math.sin(sourceAngle) * circle_radius;
                    const targetX = d.target.x + Math.cos(targetAngle) * circle_radius;
                    const targetY = d.target.y + Math.sin(targetAngle) * circle_radius;

                    // Define curvature
                    const curvature = 0.25 * (linknum - (totalLinks - 1) / 2);

                    // Calculate the path
                    const path = `M${sourceX},${sourceY}A${dr * Math.abs(curvature)},${dr * Math.abs(curvature)} 0 0,${
                        curvature > 0 ? 1 : 0
                    } ${targetX},${targetY}`;

                    return path;
                }

                function ticked() {
                    linkGroup.selectAll("g").each(function (d) {
                        const link = d3.select(this);
                        const path = linkPath(d);

                        const pathElement = link.select("path").attr("d", path).node();

                        // Ensure path element exists before calculating its length
                        if (pathElement) {
                            const totalLength = pathElement.getTotalLength();
                            const midPoint = pathElement.getPointAtLength(totalLength / 2);

                            // Update label positions
                            link.select(".label").attr("transform", `translate(${midPoint.x},${midPoint.y})`);

                            link.select(".label rect").attr("x", -25).attr("y", -15);

                            link.select(".label text").attr("x", 0).attr("y", 0);
                        }
                    });

                    nodeGroup.selectAll("g").attr("transform", (d) => `translate(${d.x},${d.y})`);
                }

                function dragstarted(event, d) {
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x;
                    d.fy = d.y;
                }

                function dragged(event, d) {
                    d.fx = event.x;
                    d.fy = event.y;
                }

                function dragended(event, d) {
                    if (!event.active) simulation.alphaTarget(0);
                    d.fx = null;
                    d.fy = null;
                }

                function getIconFromName(name) {
                    const iconMap = {
                        Staircase: "\ue289",
                        Computer: "\ue4e5",
                        Electrical: "\uf0eb",
                        Washroom: "\uf7d8",
                        Meeting: "\ue537",
                        Training: "\uf51c",
                        Hallway: "\uf557",
                        Smoking: "\uf48d",
                        Security: "\ue54a",
                        Prayer: "\uf683",
                        Mechanical: "\uf0ad",
                        Cafeteria: "\uf0f4",
                        Outside: "\uf850",
                        Loading: "\uf4de"
                    };
                    const trimmedName = name.trim().toLowerCase();
                    const matchingKeyword = Object.keys(iconMap).find((keyword) =>
                        trimmedName.includes(keyword.toLowerCase())
                    );
                    return iconMap[matchingKeyword] || "\uf0db";
                }

                update();
            });
        </script>
    </body>
</html>

"""

# COMMAND ----------

# Render the HTML content directly inside Databricks
displayHTML(html_room_path_graph_template)
