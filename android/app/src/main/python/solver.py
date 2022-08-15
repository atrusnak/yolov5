import twophase.solver as sv

def solve(colorString): 
    scrambleString = convertColorString(colorString)
    try:
        solution = sv.solve(scrambleString, 20, 5)
    except Exception as e:
        solution = "error"
        print('hi')
    return solution

def convertColorString(colorString):
    try:
        scrambleString = ""
        #UUUUUURRRRRRFFFFFFDDDDDLLLLLLLBBBBBBB
        topColor = colorString[9*0 + 4]
        rightColor = colorString[9*1 + 4]
        frontColor = colorString[9*2 + 4]
        bottomColor = colorString[9*3 + 4]
        leftColor = colorString[9*4 + 4]
        backColor = colorString[9*5 + 4]

        colorDict = {
                frontColor: 'F',
                leftColor: 'L',
                backColor: 'B',
                rightColor: 'R',
                topColor: 'U',
                bottomColor: 'D'
            }

        for c in colorString:
            scrambleString += colorDict[c]
        return scrambleString
            
    except Exception as e:
       return "asdf"  


    
