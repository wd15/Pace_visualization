import click
import pandas


@click.command()
@click.option('--filename', default='mprof.dat', help='the name of the memory file')
def run(filename):
    data = pandas.read_csv(
        filename,
        sep=' ',
        skiprows=1,
        index_col=False,
        error_bad_lines=False,
        header=None,
        warn_bad_lines=False
    ).iloc[:,1].max()
    print(data)

if __name__ == '__main__':
    run()
