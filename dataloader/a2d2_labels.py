"""
Convert between color, id, name
"""
from collections import namedtuple

__all__ = ['labels_11', 'labels_aev_38', 'map_38_to_11']

Label = namedtuple('Label', ['color', 'id', 'name'])

labels_aev_38 = [
    Label((255, 0, 0), 0, 'Car'),
    Label((200, 0, 0), 0, 'Car'),
    Label((150, 0, 0), 0, 'Car'),
    Label((128, 0, 0), 0, 'Car'),
    Label((182, 89, 6), 1, 'Bicycle'),
    Label((150, 50, 4), 1, 'Bicycle'),
    Label((90, 30, 1), 1, 'Bicycle'),
    Label((90, 30, 30), 1, 'Bicycle'),
    Label((204, 153, 255), 2, 'Pedestrian'),
    Label((189, 73, 155), 2, 'Pedestrian'),
    Label((239, 89, 191), 2, 'Pedestrian'),
    Label((255, 128, 0), 3, 'Truck'),
    Label((200, 128, 0), 3, 'Truck'),
    Label((150, 128, 0), 3, 'Truck'),
    Label((0, 255, 0), 4, 'Small vehicles'),
    Label((0, 200, 0), 4, 'Small vehicles'),
    Label((0, 150, 0), 4, 'Small vehicles'),
    Label((0, 128, 255), 5, 'Traffic signal'),
    Label((30, 28, 158), 5, 'Traffic signal'),
    Label((60, 28, 100), 5, 'Traffic signal'),
    Label((0, 255, 255), 6, 'Traffic sign'),
    Label((30, 220, 220), 6, 'Traffic sign'),
    Label((60, 157, 199), 6, 'Traffic sign'),
    Label((255, 255, 0), 7, 'Utility vehicle'),
    Label((255, 255, 200), 7, 'Utility vehicle'),
    Label((233, 100, 0), 8, 'Sidebars'),
    Label((110, 110, 0), 9, 'Speed bumper'),
    Label((128, 128, 0), 10, 'Curbstone'),
    Label((255, 193, 37), 11, 'Solid line'),
    Label((64, 0, 64), 12, 'Irrelevant signs'),
    Label((185, 122, 87), 13, 'Road blocks'),
    Label((0, 0, 100), 14, 'Tractor'),
    Label((139, 99, 108), 15, 'Non-drivable street'),
    Label((210, 50, 115), 16, 'Zebra crossing'),
    Label((255, 0, 128), 17, 'Obstacles / trash'),
    Label((255, 246, 143), 18, 'Poles'),
    Label((150, 0, 150), 19, 'RD restricted area'),
    Label((204, 255, 153), 20, 'Animals'),
    Label((238, 162, 173), 21, 'Grid structure'),
    Label((33, 44, 177), 22, 'Signal corpus'),
    Label((180, 50, 180), 23, 'Drivable cobblestone'),
    Label((255, 70, 185), 24, 'Electronic traffic'),
    Label((238, 233, 191), 25, 'Slow drive area'),
    Label((147, 253, 194), 26, 'Nature object'),
    Label((150, 150, 200), 27, 'Parking area'),
    Label((180, 150, 200), 28, 'Sidewalk'),
    Label((72, 209, 204), 29, 'Ego car'),
    Label((200, 125, 210), 30, 'Painted driv. instr.'),
    Label((159, 121, 238), 31, 'Traffic guide obj.'),
    Label((128, 0, 255), 32, 'Dashed line'),
    Label((255, 0, 255), 33, 'RD normal street'),
    Label((135, 206, 255), 34, 'Sky'),
    Label((241, 230, 255), 35, 'Buildings'),
    Label((96, 69, 143), 36, 'Blurred area'),
    Label((53, 46, 82), 37, 'Rain dirt')]

labels_aev_38_unique = [
    Label((255, 0, 0), 0, 'Car'),
    Label((182, 89, 6), 1, 'Bicycle'),
    Label((204, 153, 255), 2, 'Pedestrian'),
    Label((255, 128, 0), 3, 'Truck'),
    Label((0, 255, 0), 4, 'Small vehicles'),
    Label((0, 128, 255), 5, 'Traffic signal'),
    Label((0, 255, 255), 6, 'Traffic sign'),
    Label((255, 255, 0), 7, 'Utility vehicle'),
    Label((233, 100, 0), 8, 'Sidebars'),
    Label((110, 110, 0), 9, 'Speed bumper'),
    Label((128, 128, 0), 10, 'Curbstone'),
    Label((255, 193, 37), 11, 'Solid line'),
    Label((64, 0, 64), 12, 'Irrelevant signs'),
    Label((185, 122, 87), 13, 'Road blocks'),
    Label((0, 0, 100), 14, 'Tractor'),
    Label((139, 99, 108), 15, 'Non-drivable street'),
    Label((210, 50, 115), 16, 'Zebra crossing'),
    Label((255, 0, 128), 17, 'Obstacles / trash'),
    Label((255, 246, 143), 18, 'Poles'),
    Label((150, 0, 150), 19, 'RD restricted area'),
    Label((204, 255, 153), 20, 'Animals'),
    Label((238, 162, 173), 21, 'Grid structure'),
    Label((33, 44, 177), 22, 'Signal corpus'),
    Label((180, 50, 180), 23, 'Drivable cobblestone'),
    Label((255, 70, 185), 24, 'Electronic traffic'),
    Label((238, 233, 191), 25, 'Slow drive area'),
    Label((147, 253, 194), 26, 'Nature object'),
    Label((150, 150, 200), 27, 'Parking area'),
    Label((180, 150, 200), 28, 'Sidewalk'),
    Label((72, 209, 204), 29, 'Ego car'),
    Label((200, 125, 210), 30, 'Painted driv. instr.'),
    Label((159, 121, 238), 31, 'Traffic guide obj.'),
    Label((128, 0, 255), 32, 'Dashed line'),
    Label((255, 0, 255), 33, 'RD normal street'),
    Label((135, 206, 255), 34, 'Sky'),
    Label((241, 230, 255), 35, 'Buildings'),
    Label((96, 69, 143), 36, 'Blurred area'),
    Label((53, 46, 82), 37, 'Rain dirt')
]


map_38_to_11 = [
    (0, 1),  # Car to Car
    (1, 0),  # Bicycle to Person
    (2, 0),  # Pedestrian to Person
    (3, 2),  # Truck to Truck
    (4, 1),  # Small vehicles to Car
    (5, 6),  # Traffic signal to Info
    (6, 6),  # Traffic sign to Info
    (7, 2),  # Utility vehicle to Truck
    (8, 6),  # Sidebars to Info
    (9, 6),  # Speed bumper to Info
    (10, 4),  # Curbstone to Nondrivable
    (11, 10),  # Solid line to Lanes
    (12, 5),  # Irrelevant signs to Blocker
    (13, 5),  # Road blocks to Blocker
    (14, 2),  # Tractor to Truck
    (15, 4),  # Non-drivable street to Nondrivable
    (16, 6),  # Zebra crossing to Info
    (17, 5),  # Obstacles / trash to Blocker
    (18, 5),  # Poles to Blocker
    (19, 4),  # RD restricted area to Nondrivable
    (20, 0),  # Animals to Person
    (21, 5),  # Grid structure to Blocker
    (22, 5),  # Signal corpus to Blocker
    (23, 3),  # Drivable cobblestone to Drivable
    (24, 6),  # Electronic traffic to Info
    (25, 4),  # Slow drive area to Nondrivable
    (26, 9),  # Nature to Nature
    (27, 3),  # Parking area to Drivable
    (28, 4),  # Sidewalk to Nondrivable
    (29, 1),  # Ego car to Car
    (30, 6),  # Painted driv. instr. to Info
    (31, 6),  # Traffic guide obj. to Info
    (32, 10),  # Dashed line to Lanes
    (33, 3),  # RD normal street to Drivable
    (34, 7),  # Sky to Sky
    (35, 8),  # Buildings to Buildings
    (36, 250),  # Blurred area - images with this class will be ignored
    (37, 250),  # Rain dirt - images with this class will be ignored
    # (36, 0),  # Blurred area - images with this class will be ignored
    # (37, 0),  # Rain dirt - images with this class will be ignored
    ]

# Defintion of the 11 classes
labels_11 = [
    Label((255, 0, 121), 0, 'Person'),
    Label((255, 15, 15), 1, 'Car'),
    Label((254, 83, 1), 2, 'Truck'),
    Label((0, 255, 0), 3, 'Drivable'),
    Label((255, 255, 0), 4, 'Nondrivable'),
    Label((192, 192, 192), 5, 'Blocker'),
    Label((0, 0, 255), 6, 'Info'),
    Label((128, 255, 255), 7, 'Sky'),
    Label((83, 0, 0), 8, 'Buildings'),
    Label((0, 80, 0), 9, 'Nature'),
    Label((128, 0, 255), 10, 'Lanes')
]


replacement_aev_11 = dict(
    classes=11,
    class_mapping='11_classes_aev',
    labels_IDtoRGB=labels_11
)
