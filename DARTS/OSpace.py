from Cells import EDGE_PARAMETERS, CELL_PARAMETERS


def generateOSpaceCell():
    parameters = {}

    for key, params in EDGE_PARAMETERS.items():
        parameters[key] = list(params)

    return generateO(parameters)


def generateOSpaceCellChannels():
    parameters = {}
    for key, params in CELL_PARAMETERS.items():
        parameters[key] = list(params)

    return generateO(parameters)



def generateO(parameters):
    O = [f"{k}${value}$" for k in list(parameters.keys())[:1] for value in parameters[k]]

    for key in list(parameters.keys())[1:]:
        list1 = list(O)
        O = []
        list2 = parameters[key]
        for i in range(len(list1)):
            if key == "channels" and "operation$3" not in list1[i]:
                O.append(list1[i])
            elif "operation$0" in list1[i]:
                O.append(list1[i])
            elif "operation$4" in list1[i]:
                O.append(list1[i])
            else:
                for j in range(len(list2)):
                    O.append(f"{list1[i]}{key}${list2[j]}$")
    return O


def main():
    generateOSpaceCell()


if __name__ == "__main__":
    main()
