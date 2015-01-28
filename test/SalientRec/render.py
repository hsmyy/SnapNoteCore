from jinja2 import Template
import os

def renderHtml(input, cross, turned):
    files = os.listdir(input)
    resultList = [{'origin' : input + f, 'seg' : cross + f, 'output': turned + f} for f in files]

    with open("mytemplate.html","r") as input:
        template = Template( ''.join( input.readlines() ) )
        with open("testResult.html","w") as output:
            output.write(template.render(result=resultList))

if __name__ == '__main__':
    renderHtml('test/input/', 'test/seg/','test/output/')
