import click

@click.command()
@click.option('-n','--name', prompt='Your name', help='The person to greet.')
def hello(name):
    """Simple program that greets NAME"""
    click.echo('Hello %s!' % name)

if __name__ == '__main__':
    hello()
    
