import cPickle
import logging
import os
import xml.etree.ElementTree as etree
import numpy as np

from nodule_structs import RadAnnotation, SmallNodule, NormalNodule, \
    NoduleRoi, NonNodule, AnnotationHeader

NS = {'nih': 'http://www.nih.gov'}


def find_all_files(root, suffix=None):
    res = []
    for root, _, files in os.walk(root):
        for f in files:
            if suffix is not None and not f.endswith(suffix):
                continue
            res.append(os.path.join(root, f))
    return res


def parse_dir(dirname, outdir, flatten=True, pickle=True):
    assert os.path.isdir(dirname)

    if not flatten:
        return parse_original_xmls(dirname, outdir, pickle)

    pickle_file = os.path.join(outdir, 'annotation_flatten.pkl')
    if os.path.isfile(pickle_file):
        logging.info("Loading annotations from file %s" % pickle_file)
        with open(pickle_file, 'r') as f:
            annotations = cPickle.load(f)
        logging.info("Load annotations complete")
        return annotations
    annotations = parse_original_xmls(dirname, outdir, pickle)
    annotations = flatten_annotation(annotations)
    if pickle:
        logging.info("Saving annotations to file %s" % pickle_file)
        with open(pickle_file, 'w') as f:
            cPickle.dump(annotations, f)
    return annotations


def parse_original_xmls(dirname, outdir, pickle=True):
    pickle_file = pickle and os.path.join(outdir, 'annotation.pkl') or None
    if pickle and os.path.isfile(pickle_file):
        logging.info("Loading annotations from file %s" % pickle_file)
        with open(pickle_file, 'r') as f:
            annotations = cPickle.load(f)
        logging.info("Load annotations complete")
    else:
        logging.info("Reading annotations")
        annotations = []
        xml_files = find_all_files(dirname, '.xml')
        for f in xml_files:
            annotations.append(parse(f))
    if pickle and not os.path.isfile(pickle_file):
        logging.info("Saving annotations to file %s" % pickle_file)
        with open(pickle_file, 'w') as f:
            cPickle.dump(annotations, f)
    return annotations


def parse(xml_filename):
    logging.info("Parsing %s" % xml_filename)
    annotations = []
    # ET is the library we use to parse xml data
    tree = etree.parse(xml_filename)
    root = tree.getroot()
    # header = parse_header(root)
    # readingSession-> holds radiologist's annotation info
    for read_session in root.findall('nih:readingSession', NS):
        # to hold each radiologists annotation
        # i.e. readingSession in xml file
        rad_annotation = RadAnnotation()
        rad_annotation.version = \
            read_session.find('nih:annotationVersion', NS).text
        rad_annotation.id = \
            read_session.find('nih:servicingRadiologistID', NS).text

        # nodules
        nodule_nodes = read_session.findall('nih:unblindedReadNodule', NS)
        for node in nodule_nodes:
            nodule = parse_nodule(node)
            if nodule.is_small:
                rad_annotation.small_nodules.append(nodule)
            else:
                rad_annotation.nodules.append(nodule)

        # non-nodules
        non_nodule = read_session.findall('nih:nonNodule', NS)
        for node in non_nodule:
            nodule = parse_non_nodule(node)
            rad_annotation.non_nodules.append(nodule)
        annotations.append(rad_annotation)
    return annotations


def parse_header(root):
    header = AnnotationHeader()
    print root.findall('nih:*', NS)
    resp_hdr = root.findall('nih:ResponseHeader', NS)[0]
    header.version = resp_hdr.find('nih:Version', NS).text
    header.message_id = resp_hdr.find('nih:MessageId', NS).text
    header.date_request = resp_hdr.find('nih:DateRequest', NS).text
    header.time_request = resp_hdr.find('nih:TimeRequest', NS).text
    header.task_desc = resp_hdr.find('nih:TaskDescription', NS).text
    header.series_instance_uid = resp_hdr.find('nih:SeriesInstanceUid', NS).text
    date_service = resp_hdr.find('nih:DateService', NS)
    if date_service is not None:
        header.date_service = date_service.text
    time_service = resp_hdr.find('nih:TimeService', NS)
    if time_service is not None:
        header.time_service = time_service.text
    header.study_instance_uid = resp_hdr.find('nih:StudyInstanceUID', NS).text
    return header


def parse_nodule(xml_node):  # xml_node is one unblindedReadNodule
    char_node = xml_node.find('nih:characteristics', NS)
    # if no characteristics, it is smallnodule  i.e. is_small=TRUE
    is_small = (char_node is None or len(char_node) == 0)
    nodule = is_small and SmallNodule() or NormalNodule()
    nodule.id = xml_node.find('nih:noduleID', NS).text
    if not is_small:
        subtlety = char_node.find('nih:subtlety', NS)
        nodule.characteristics.subtlety = int(subtlety.text)
        nodule.characteristics.internal_struct = \
            int(char_node.find('nih:internalStructure', NS).text)
        nodule.characteristics.calcification = \
            int(char_node.find('nih:calcification', NS).text)
        nodule.characteristics.sphericity = \
            int(char_node.find('nih:sphericity', NS).text)
        nodule.characteristics.margin = \
            int(char_node.find('nih:margin', NS).text)
        nodule.characteristics.lobulation = \
            int(char_node.find('nih:lobulation', NS).text)
        nodule.characteristics.spiculation = \
            int(char_node.find('nih:spiculation', NS).text)
        nodule.characteristics.texture = \
            int(char_node.find('nih:texture', NS).text)
        nodule.characteristics.malignancy = \
            int(char_node.find('nih:malignancy', NS).text)
    xml_rois = xml_node.findall('nih:roi', NS)
    for xml_roi in xml_rois:
        roi = NoduleRoi()
        roi.z = float(xml_roi.find('nih:imageZposition', NS).text)
        roi.sop_uid = xml_roi.find('nih:imageSOP_UID', NS).text
        # when inclusion = TRUE ->roi includes the whole nodule
        # when inclusion = FALSE ->roi is drown twice for one nodule
        # 1.ouside the nodule
        # 2.inside the nodule -> to indicate that the nodule has donut
        # hole(the inside hole is
        # not part of the nodule) but by forcing inclusion to be TRUE,
        # this situation is ignored
        roi.inclusion = (xml_roi.find('nih:inclusion', NS).text == "TRUE")
        edge_maps = xml_roi.findall('nih:edgeMap', NS)
        for edge_map in edge_maps:
            x = int(edge_map.find('nih:xCoord', NS).text)
            y = int(edge_map.find('nih:yCoord', NS).text)
            roi.roi_xy.append([x, y])
        xmax = np.array(roi.roi_xy)[:, 0].max()
        xmin = np.array(roi.roi_xy)[:, 0].min()
        ymax = np.array(roi.roi_xy)[:, 1].max()
        ymin = np.array(roi.roi_xy)[:, 1].min()
        if not is_small:  # only for normalNodules
            roi.roi_rect = (xmin, ymin, xmax, ymax)
            roi.roi_centroid = (
                (xmax + xmin) / 2., (ymin + ymax) / 2.)  # center point
        nodule.rois.append(roi)
    return nodule  # is equivalent to unblindedReadNodule(xml element)


def parse_non_nodule(xml_node):  # xml_node is one nonNodule
    nodule = NonNodule()
    nodule.id = xml_node.find('nih:nonNoduleID', NS).text
    roi = NoduleRoi()
    roi.z = float(xml_node.find('nih:imageZposition', NS).text)
    roi.sop_uid = xml_node.find('nih:imageSOP_UID', NS).text
    loci = xml_node.findall('nih:locus', NS)
    for locus in loci:
        x = int(locus.find('nih:xCoord', NS).text)
        y = int(locus.find('nih:yCoord', NS).text)
        roi.roi_xy.append((x, y))
    nodule.rois.append(roi)
    return nodule  # is equivalent to nonNodule(xml element)


def flatten_annotation(annotation_dict):
    logging.info("Start flatten")
    # res = {}
    res = {'nodules': [], 'small_nodules': [], 'non_nodules': []}
    for annotations in annotation_dict:
        # annotations in each file
        for anno in annotations:
            flatten_nodule(anno.nodules, 'nodules', res)
            flatten_nodule(anno.small_nodules, 'small_nodules', res)
            flatten_nodule(anno.non_nodules, 'non_nodules', res)
    logging.info("Flatten complete")
    return res


def flatten_nodule(nodules, type, result):
    # if not result:
    #     result = {'nodules': [], 'small_nodules': [], 'non_nodules': []}
    for nodule in nodules:
        point = []
        for roi in nodule.rois:
            tmp = {'pixels': roi.roi_xy, 'sop_uid': roi.sop_uid}
            point.append(tmp)

        result[type].append(point)

if __name__ == '__main__':
    parse_dir('./data/LIDC-IDRI-0001/')
