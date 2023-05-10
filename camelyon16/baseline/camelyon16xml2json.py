import sys
import os
import argparse
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from camelyon16.data.annotation import Formatter  # noqa

parser = argparse.ArgumentParser(description='Convert Camelyon16 xml format to'
                                 'internal json format')
parser.add_argument('xml_path', default=None, metavar='XML_PATH', type=str,
                    help='Path to the input Camelyon16 xml annotation')
parser.add_argument('json_path', default=None, metavar='JSON_PATH', type=str,
                    help='Path to the output annotation in json format')


def run(args):
    dir = os.listdir(args.xml_path)
    for file in sorted(dir):
        if file.split('.')[-1] == 'xml':
            xml_path = os.path.join(args.xml_path, file)
            json_path = os.path.join(args.json_path, file.split('.')[0] + '.json')
            Formatter.camelyon16xml2json(xml_path, json_path)


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args([
        "/media/ps/passport2/hhy/camelyon16/training/annotations/xml",
        "/media/ps/passport2/hhy/camelyon16/training/annotations/json"])
    run(args)


if __name__ == '__main__':
    main()