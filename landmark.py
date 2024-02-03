class Landmark:
    def __init__(self) -> None:
        self.wrist = []
        self.thumb = dict(cmc=[], mcp=[], ip=[], tip=[])
        self.index_finger = dict(mcp=[], pip=[], dip=[], tip=[])
        self.middle_finger = dict(mcp=[], pip=[], dip=[], tip=[])
        self.ring_finger = dict(mcp=[], pip=[], dip=[], tip=[])
        self.pingky = dict(mcp=[], pip=[], dip=[], tip=[])
        self.list_of_landmarks = []
        self.list_of_landmarks_xy = []
        self.size = 0
        self.all_x = []
        self.all_y = []
        self.xyxynormalize = []
        self.xywhnormalize = []
        self.bbox = []

    def set_landmarks(self, results):
        self.set_wrist(results)
        self.set_thumb(results)
        self.set_index_finger(results)
        self.set_middle_finger(results)
        self.set_ring_finger(results)
        self.set_pinky(results)
        self.set_list_of_landmarks()
        self.set_all_x_and_y()
        self.set_list_of_landmarks_xy()
        

    def set_wrist(self, results):
        self.wrist = self.set_list_coordinates_landmarks(results, 0)
    
    def set_thumb(self, results):
        self.thumb['cmc'] = self.set_list_coordinates_landmarks(results, 1)
        self.thumb['mcp'] = self.set_list_coordinates_landmarks(results, 2)
        self.thumb['ip'] = self.set_list_coordinates_landmarks(results, 3)
        self.thumb['tip'] = self.set_list_coordinates_landmarks(results, 4)
        
    def set_index_finger(self, results):
        self.index_finger['mcp'] = self.set_list_coordinates_landmarks(results, 5)
        self.index_finger['pip'] = self.set_list_coordinates_landmarks(results, 6)
        self.index_finger['dip'] = self.set_list_coordinates_landmarks(results, 7)
        self.index_finger['tip'] = self.set_list_coordinates_landmarks(results, 8)

    def set_middle_finger(self, results):
        self.middle_finger['mcp'] = self.set_list_coordinates_landmarks(results, 9)
        self.middle_finger['pip'] = self.set_list_coordinates_landmarks(results, 10)
        self.middle_finger['dip'] = self.set_list_coordinates_landmarks(results, 11)
        self.middle_finger['tip'] = self.set_list_coordinates_landmarks(results, 12)


    def set_ring_finger(self, results):
        self.ring_finger['mcp'] = self.set_list_coordinates_landmarks(results, 13)
        self.ring_finger['pip'] = self.set_list_coordinates_landmarks(results, 14)
        self.ring_finger['dip'] = self.set_list_coordinates_landmarks(results, 15)
        self.ring_finger['tip'] = self.set_list_coordinates_landmarks(results, 16)

    def set_pinky(self, results):
        self.pingky['mcp'] = self.set_list_coordinates_landmarks(results, 17)
        self.pingky['pip'] = self.set_list_coordinates_landmarks(results, 18)
        self.pingky['dip'] = self.set_list_coordinates_landmarks(results, 19)
        self.pingky['tip'] = self.set_list_coordinates_landmarks(results, 20)

    def set_list_coordinates_landmarks(self,results, index):
        list = []
        list.append(results[index].x)
        list.append(results[index].y)
        list.append(results[index].z)

        return list
    

    def set_list_of_landmarks(self):
        landmark_list = []
        landmark_list.append(self.wrist)
        landmark_list.append(self.thumb['cmc'])
        landmark_list.append(self.thumb['mcp'])
        landmark_list.append(self.thumb['ip'])
        landmark_list.append(self.thumb['tip'])
        landmark_list.append(self.index_finger['mcp'])
        landmark_list.append(self.index_finger['pip'])
        landmark_list.append(self.index_finger['dip'])
        landmark_list.append(self.index_finger['tip'])
        landmark_list.append(self.middle_finger['mcp'])
        landmark_list.append(self.middle_finger['pip'])
        landmark_list.append(self.middle_finger['dip'])
        landmark_list.append(self.middle_finger['tip'])
        landmark_list.append(self.ring_finger['mcp'])
        landmark_list.append(self.ring_finger['pip'])
        landmark_list.append(self.ring_finger['dip'])
        landmark_list.append(self.ring_finger['tip'])
        landmark_list.append(self.pingky['mcp'])
        landmark_list.append(self.pingky['pip'])
        landmark_list.append(self.pingky['dip'])
        landmark_list.append(self.pingky['tip'])

        self.list_of_landmarks = landmark_list

    def set_list_of_landmarks_xy(self):
        list_xy_only = [[x, y] for x, y, z in self.list_of_landmarks]
        self.list_of_landmarks_xy = list_xy_only


    def set_all_x_and_y(self):
        self.all_x = [coordinates[0] for coordinates in self.list_of_landmarks]
        self.all_y = [coordinates[1] for coordinates in self.list_of_landmarks]

    def get_bbox_coordinates(self, image_shape):
        all_x = [x * image_shape[1] for x in self.all_x]
        all_y = [y * image_shape[0] for y in self.all_y]
        
        self.bbox = min(all_x), min(all_y), max(all_x), max(all_y)

        return self.bbox # return as (xmin, ymin, xmax, ymax)
    
    def get_bbox_normalized(self, image_shape):
        xmin,ymin,xmax,ymax = self.bbox
        
        xyxy = []
        xyxy.append(xmin / image_shape[0])
        xyxy.append(ymin / image_shape[0])
        xyxy.append(xmax / image_shape[1])
        xyxy.append(ymax / image_shape[1])

        self.xyxynormalize = xyxy

        return self.xyxynormalize
    def get_bbox_coordinates_xywh(self, image_shape):
        xmin,ymin,xmax,ymax = self.bbox
        center_x = (xmin + xmax) // 2
        center_y = (ymin + ymax) // 2

        return center_x / image_shape[0], center_y / image_shape[1], (xmax - xmin) / image_shape[0], (ymax - ymin) / image_shape[1]
    
    

