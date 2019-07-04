#!/usr/bin/env python
"""
Quickly renders a single jinja2 template file from command line.
"""

import logging
logger = logging.getLogger(__name__)
logfmt = "[%(levelname)s - %(filename)s:%(lineno)s - %(funcName)s() ] %(message)s (%(asctime)s)"
logging.basicConfig( format = logfmt )

def render(
  infile,
  outfile,
  context = {'machine':'NEMO', 'mode':'PRODUCTION'} ):
    """Renders jinja2 template file.

    Example:

    Args:
        infile (str): .yaml emplate file.
        outfile (str):  rendered .yaml file.
        context (str or dict): jinja2 context, YAML format if str.

    Returns:
        Nothing.
    """
    from jinja2 import Template
    # Environment, FileSystemLoader

    #env = Environment(
    #      loader=FileSystemLoader(self.template_dir),
    #      autoescape = False)

    if isinstance(context, str):
        import yaml
        context_str = context
        context = yaml.safe_load(context_str)
        logger.debug("Parsed '{}' as YAML '{}'".format(context_str, context))

    with open(infile) as template_file:
        template = Template(template_file.read())

    rendered = template.render(context)
    logger.debug("Rendered infile as \n{}".format(rendered))

    with open(outfile, "w") as rendered_file:
        rendered_file.write(rendered)

    return

def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('infile',
        help='Template .yaml input file')
    parser.add_argument('outfile',
        help='Rendered .yaml output file')
    parser.add_argument('--context',
        help="Context",
        default={'machine': 'NEMO', 'mode': 'PRODUCTION'})
    parser.add_argument('--verbose', '-v', action='store_true',
        help='Make this tool more verbose')
    parser.add_argument('--debug', action='store_true',
        help='Make this tool print debug info')
    args = parser.parse_args()

    if args.debug:
        loglevel = logging.DEBUG
    elif args.verbose:
        loglevel = logging.INFO
    else:
        loglevel = logging.WARNING

    logger.setLevel(loglevel)

    logger.debug( args )

    logger.info("Render template {} to output file {} with context {}".format(
        args.infile,
        args.outfile,
        args.context ) )
    render(
        args.infile,
        args.outfile,
        args.context )

if __name__ == '__main__':
    main()
