label_to_kwh_brussels = {
    "A": 75,
    "B": 125,
    "C": 175,
    "D": 225,
    "E": 325,
    "F": 425,
    "G": 525,
}

label_to_kwh_wallonia = {
    "A+": 45,
    "A": 95,
    "B": 150,
    "C": 225,
    "D": 300,
    "E": 400,
    "F": 475,
    "G": 550,
}

label_to_kwh_flanders = {
    "A+": -50,
    "A": 75,
    "B": 150,
    "C": 225,
    "D": 300,
    "E": 400,
    "F": 500,
}

def map_label_to_kwh(row):
    """
    Maps the EPC label to kWh based on the region.
    """

    region = row["province"]
    label = row["epcScore"]
    
    if region.upper() == "BRUSSELS":
        return label_to_kwh_brussels.get(label, None)
    elif region.upper() in ['NAMUR', 'LIEGE', 'HAINAUT', 'BRABANT WALLON', 'LUXEMBOURG']:
        return label_to_kwh_wallonia.get(label, None)
    elif region.upper() in ['ANTWERP', 'WEST FLANDERS', 'EAST FLANDERS', 'FLEMISH BRABANT', 'LIMBURG']:
        return label_to_kwh_flanders.get(label, None)
    else:
        return None