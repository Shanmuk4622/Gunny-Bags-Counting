import math
from collections import deque


class Tracker:
    def __init__(self):
        # Store the center positions of the objects (IDs)
        self.center_points = {}
        # Keep the count of the IDs
        self.id_count = 0
        # Maximum number of frames an object can be lost before removal
        self.max_lost_frames = 30
        self.lost_frames = {}

    def update(self, objects_rect):
        objects_bbs_ids = []
        # Keep track of the objects that are still valid
        current_frame_ids = []

        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Flag to check if object already detected
            same_object_detected = False

            # Check if this object was detected already (using center point comparison)
            for obj_id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:  # Threshold distance for matching
                    self.center_points[obj_id] = (cx, cy)  # Update position
                    self.lost_frames[obj_id] = 0  # Reset lost frames counter
                    objects_bbs_ids.append([x, y, w, h, obj_id])
                    current_frame_ids.append(obj_id)  # Object is valid this frame
                    same_object_detected = True
                    break

            # If object not detected, create a new ID
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                self.lost_frames[self.id_count] = 0  # Track it as a new object
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                current_frame_ids.append(self.id_count)
                self.id_count += 1

        # Remove lost objects: track only those detected recently
        objects_bbs_ids = [obj for obj in objects_bbs_ids if obj[4] in current_frame_ids]

        # Increment lost frames for objects not tracked in this frame
        for obj_id in list(self.center_points.keys()):
            if obj_id not in current_frame_ids:
                self.lost_frames[obj_id] += 1
                if self.lost_frames[obj_id] > self.max_lost_frames:
                    del self.center_points[obj_id]  # Remove object if too many frames lost
                    del self.lost_frames[obj_id]  # Remove its lost frame tracking

        return objects_bbs_ids
