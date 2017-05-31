"""
Modified from https://github.com/tizita-nesibu/lidc-idri-visualization
"""


class NoduleCharstics:
    def __init__(self):
        self.subtlety = 0
        self.internal_struct = 0
        self.calcification = 0
        self.sphericity = 0
        self.margin = 0
        self.lobulation = 0
        self.spiculation = 0
        self.texture = 0
        self.malignancy = 0
        return

    def __str__(self):
        str = "subtlty (%d) intstruct (%d) calci (%d) sphere (%d) " \
              "margin (%d) lob (%d) spicul (%d) txtur (%d) malig (%d)" % (
                  self.subtlety, self.internal_struct, self.calcification,
                  self.sphericity,
                  self.margin, self.lobulation, self.spiculation, self.texture,
                  self.malignancy)
        return str

    def set_values(self, sub, inter, calc, spher, lob, spic, tex, malig):
        self.subtlety = sub
        self.internal_struct = inter
        self.calcification = calc
        self.sphericity = spher
        self.lobulation = lob
        self.spiculation = spic
        self.texture = tex
        self.malignancy = malig
        return


class NoduleRoi:  # is common for nodule and non-nodule
    def __init__(self, z_pos=0., sop_uid=''):
        self.z = z_pos
        self.sop_uid = sop_uid
        self.inclusion = True

        self.roi_xy = []  # to hold list of x,ycords in edgemap(edgmap pairs)
        self.roi_rect = []  # rectangle to hold the roi
        self.roi_centroid = []  # to hold centroid of the roi
        return

    def __str__(self):
        n_pts = len(self.roi_xy)
        str = "Inclusion (%s) Z = %.2f SOP_UID (%s) \n ROI points [ %d ]  ::  " \
              "" % (
            self.inclusion, self.z, self.sop_uid, n_pts)

        if (n_pts > 2):
            str += "[[ %d,%d ]] :: " % (
            self.roi_centroid[0], self.roi_centroid[1])
            str += "(%d, %d), (%d,%d)..." % (
                self.roi_xy[0][0], self.roi_xy[0][1], self.roi_xy[1][0],
                self.roi_xy[1][1])
            str += "(%d, %d), (%d,%d)" % (
                self.roi_xy[-2][0], self.roi_xy[-2][1], self.roi_xy[-1][0],
                self.roi_xy[-1][1])
        else:
            for i in range(n_pts):
                str += "(%d, %d)," % (self.roi_xy[i][0], self.roi_xy[i][1])
        return str


class Nodule:  # is base class for all nodule types (NormalNodule,
    # SmallNodule, NonNodule)
    def __init__(self):
        self.id = None
        self.rois = []
        self.is_small = False

    def __str__(self):
        strng = "--- Nodule ID (%s) Small [%s] ---\n" % (
        self.id, str(self.is_small))
        strng += self.tostring() + "\n"
        return strng

    def tostring(self):
        pass


class NoduleAnnotationCluster():  # to be seen
    def __init__(self):
        self.id = []
        self.z_pos = []
        self.centroid = []  # (x,y) of the centroid
        #  convex hull description
        #   p0 ---- p1
        #   |       |
        #   p2-----p3
        self.convex_hull = []  # [()_0 ()_1 ()_2 ()_3]
        self.convex_hull_with_margin = []
        self.no_annots = 0
        self.nodules_data = []

    def compute_centroid(self):
        self.set_convex_hull()
        xc = 0.5 * (
        self.convex_hull[0][0] + self.convex_hull[3][0])  # (x_min + x_max)/2
        yc = 0.5 * (
        self.convex_hull[0][1] + self.convex_hull[3][1])  # (y_min + y_max)/2
        self.centroid = (xc, yc)
        return self.centroid

    def set_convex_hull(self):
        x_min, x_max = 640, 0
        y_min, y_max = 640, 0

        for nodule in self.nodules_data:
            for roi in nodule.rois:
                for dt_pt in roi.roi_xy:
                    # roi.roi_xy -> [(x,y)]
                    # TODO : finish this loop  #?????????????????????????????
                    x_min = dt_pt[0] if (x_min > dt_pt[0]) else x_min
                    x_max = dt_pt[0] if (x_max < dt_pt[0]) else x_max
                    y_min = dt_pt[1] if (y_min > dt_pt[1]) else y_min
                    y_max = dt_pt[1] if (y_max < dt_pt[1]) else y_max
        self.convex_hull = [(x_min, y_min), (x_max, y_min), (x_min, y_max),
                            (x_max, y_max)]
        w, h = (x_max - x_min), (y_max - y_min)
        x_min = int(x_min - 0.15 * w)
        x_max = int(x_max + 0.15 * w)
        y_min = int(y_min - 0.15 * h)
        y_max = int(y_max + 0.15 * h)
        self.convex_hull_with_margin = [(x_min, y_min), (x_max, y_min),
                                        (x_min, y_max),
                                        (x_max, y_max)]


class NormalNodule(Nodule):
    def __init__(self):
        Nodule.__init__(self)
        self.characteristics = NoduleCharstics()
        self.is_small = False

    def tostring(self):
        strng = str(self.characteristics)
        strng += "\n"

        for roi in self.rois:
            strng += str(
                roi) + "\n"  # str calls __str__ of NoduleRoi's class
            # i.e.converting roi to
        return strng  # string to prepare it for printing(it doesn't print it)


class SmallNodule(Nodule):
    def __init__(self):
        Nodule.__init__(self)
        self.is_small = True

    def tostring(self):
        strng = ''
        for roi in self.rois:
            strng += str(roi) + "\n"
        return strng


class NonNodule(Nodule):
    def __init__(self):
        Nodule.__init__(self)
        self.is_small = True

    def tostring(self):
        strng = ''
        for roi in self.rois:
            strng += str(roi)
        return strng


class RadAnnotation:
    def __init__(self, init=True):
        self.version = None
        self.id = None

        self.nodules = []  # is normalNodule i.e in xml unblindedReadNodule
        # with characteristics info
        self.small_nodules = []  # in xml unblindedReadNodule with no
        # characteristics info
        self.non_nodules = []  # located inside readingSession
        self.initialized = init
        return

    def is_init(self):
        return self.initialized

    def set_init(self, init):
        self.initialized = init
        return

    def __str__(self):
        n_nodules = len(self.nodules)
        n_small_nodules = len(self.small_nodules)
        n_non_nodules = len(self.non_nodules)
        strng = "Annotation Version [%s] Radiologist ID [%s] \n" % (
        self.version, self.id)
        strng += "#Nodules [%d] #SmallNodules [%d] #NonNodules[%d] \n" % (
            n_nodules, n_small_nodules, n_non_nodules)

        if (n_nodules > 0):
            strng += "--- Nodules [%d]---\n" % n_nodules
            for i in range(n_nodules):
                strng += str(self.nodules[i])

        if (n_small_nodules > 0):
            strng += "--- Small Nodules [%d] ---\n" % n_small_nodules
            for i in range(n_small_nodules):
                strng += str(self.small_nodules[i])

        if (n_non_nodules > 0):
            strng += "--- Non Nodules [%d] ---\n" % n_non_nodules
            for i in range(n_non_nodules):
                strng += str(self.non_nodules[i])

        strng += "-" * 79 + "\n"
        return strng


class AnnotationHeader:
    def __init__(
            self):  # 4 elements are not included b/c they don't have data
        # inside
        self.version = None
        self.message_id = None
        self.date_request = None
        self.time_request = None
        self.task_desc = None
        self.series_instance_uid = None
        self.date_service = None
        self.time_service = None
        self.study_instance_uid = None

    def __str__(self):
        str = ("--- XML HEADER ---\n"
               "Version (%s) Message-Id (%s) Date-request (%s) Time-request ("
               "%s) \n"
               "Series-UID (%s)\n"
               "Time-service (%s) Task-descr (%s) Date-service (%s) "
               "Time-service (%s)\n"
               "Study-UID (%s)") % (
                  self.version, self.message_id, self.date_request,
                  self.time_request,
                  self.series_instance_uid, self.time_service, self.task_desc,
                  self.date_service,
                  self.time_service, self.study_instance_uid)
        return str


class IdriReadMessage:
    def __init__(self):
        self.header = AnnotationHeader()
        self.annotations = []
