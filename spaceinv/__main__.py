# -*- coding:utf-8 -*-

import click
from spaceinv.run import run


@click.group()
def main():
    pass


main.add_command(run, 'run')


if __name__ == '__main__':
    main()

