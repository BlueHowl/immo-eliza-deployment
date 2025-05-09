import gradio as gr
import xgboost as xgb
import pandas as pd
from pipeline.pipeline import prepare_dict
from pipeline.util.postal_code_price import postal_code_prices

# Load your XGBoost model
model = xgb.Booster()
model.load_model("models/xgb_model.json")

# user features
FEATURES = [
    "habitableSurface",
    "toiletCount",
    "province",
    "postCode",
    "bedroomCount",
    "subtype",
    "kitchenType",
    "buildingCondition",
    "landSurface",
    "hasOffice",
    "hasSwimmingPool",
    "epcScore",
    "facedeCount",
    "terraceSurface",
    "parkingCountOutdoor",
    "hasFireplace",
    "hasDressingRoom",
    "hasPhotovoltaicPanels",
    "hasHeatPump",
    "hasThermicPanels",
    "buildingConstructionYear",
]

# Default values for all pipeline fields
def create_default_dict():
    return {
        "Unnamed: 0": 0,
        "id": "default_id",
        "url": "default_url",
        "type": "default_type",
        "subtype": "APARTMENT",
        "bedroomCount": 0,
        "bathroomCount": 0,
        "province": "default_province",
        "locality": "default_locality",
        "postCode": 0,
        "habitableSurface": 0,
        "roomCount": 0,
        "monthlyCost": 0,
        "hasAttic": False,
        "hasBasement": False,
        "hasDressingRoom": False,
        "diningRoomSurface": 0,
        "hasDiningRoom": False,
        "buildingCondition": "GOOD",
        "buildingConstructionYear": 0,
        "facedeCount": 0,
        "floorCount": 0,
        "streetFacadeWidth": 0,
        "hasLift": False,
        "floodZoneType": "default_floodZoneType",
        "heatingType": "default_heatingType",
        "hasHeatPump": False,
        "hasPhotovoltaicPanels": False,
        "hasThermicPanels": False,
        "kitchenSurface": 0,
        "kitchenType": "NOT_INSTALLED",
        "landSurface": 0,
        "hasLivingRoom": False,
        "livingRoomSurface": 0,
        "hasBalcony": False,
        "hasGarden": False,
        "gardenSurface": 0,
        "gardenOrientation": "default_orientation",
        "parkingCountIndoor": 0,
        "parkingCountOutdoor": 0,
        "hasAirConditioning": False,
        "hasArmoredDoor": False,
        "hasVisiophone": False,
        "hasOffice": False,
        "toiletCount": 0,
        "hasSwimmingPool": False,
        "hasFireplace": False,
        "hasTerrace": False,
        "terraceSurface": 0,
        "terraceOrientation": "default_orientation",
        "accessibleDisabledPeople": False,
        "epcScore": "A",  # This will be transformed to epc_kwh in the pipeline
    }

def predict(input_dict):
    # Create default dict and update with user inputs
    full_dict = create_default_dict()
    full_dict.update(input_dict)
    
    # Process through pipeline
    df = prepare_dict(full_dict)
    
    # Ensure the DataFrame has exactly the columns needed by the model in the exact order
    # This is crucial to avoid feature_names mismatch error
    expected_features = ["habitableSurface","toiletCount","postCode","bedroomCount","subtype","kitchenType","buildingCondition","landSurface","hasOffice","hasSwimmingPool","epc_kwh","facedeCount","parkingCountOutdoor","hasFireplace","terraceSurface","hasPhotovoltaicPanels","hasDressingRoom","hasHeatPump","hasThermicPanels","buildingConstructionYear"]
    
    # Ensure all expected features exist in the DataFrame
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0  # Add missing features with a default value
    
    # Reorder columns to match exactly what the model expects
    df = df[expected_features]
    
    # Create DMatrix for XGBoost prediction with feature_names explicitly specified
    dmatrix = xgb.DMatrix(df, feature_names=expected_features)
    prediction = model.predict(dmatrix)
    return float(prediction[0])

# Define Gradio input components based on XGB features
inputs = [
    gr.Dropdown(['APARTMENT', 'HOUSE', 'FLAT_STUDIO', 'DUPLEX', 'PENTHOUSE',
       'APARTMENT_GROUP', 'GROUND_FLOOR', 'APARTMENT_BLOCK', 'MANSION',
       'EXCEPTIONAL_PROPERTY', 'MIXED_USE_BUILDING', 'TRIPLEX', 'LOFT',
       'VILLA', 'TOWN_HOUSE', 'CHALET', 'HOUSE_GROUP', 'MANOR_HOUSE',
       'SERVICE_FLAT', 'KOT', 'FARMHOUSE', 'BUNGALOW', 'COUNTRY_COTTAGE',
       'OTHER_PROPERTY', 'CASTLE', 'PAVILION'], label="Property Subtype"),
    gr.Number(label="Bedroom Count", precision=0, minimum=0),
    gr.Dropdown(["BRUSSELS", 'NAMUR', 'LIEGE', 'HAINAUT', 'BRABANT WALLON', 'LUXEMBOURG', 'ANTWERP', 'WEST FLANDERS', 'EAST FLANDERS', 'FLEMISH BRABANT', 'LIMBURG'], label="Province"),
    # gr.Number(label="Post Code", precision=0),
    gr.Dropdown(postal_code_prices.keys(), label="Postal Code"),
    gr.Number(label="Habitable Surface (m²)", precision=0, minimum=0),
    gr.Checkbox(label="Has Dressing Room"),
    gr.Dropdown(['GOOD', 'TO_BE_DONE_UP', 'AS_NEW', 'JUST_RENOVATED', 'TO_RENOVATE', 'TO_RESTORE'], label="Building Condition"),
    gr.Number(label="Construction Year", precision=0, minimum=1000, value=2000),
    gr.Number(label="Facade Count", precision=0, minimum=1, maximum=4, value=2),
    gr.Checkbox(label="Has Heat Pump"),
    gr.Checkbox(label="Has Photovoltaic Panels"),
    gr.Checkbox(label="Has Thermic Panels"),
    gr.Dropdown(['SEMI_EQUIPPED', 'INSTALLED', 'HYPER_EQUIPPED', 'NOT_INSTALLED', 'USA_UNINSTALLED', 'USA_HYPER_EQUIPPED', 'USA_INSTALLED', 'USA_SEMI_EQUIPPED'], label="Kitchen Type"),
    gr.Number(label="Land Surface (m²)", precision=0, minimum=0),
    gr.Number(label="Outdoor Parking Count", precision=0, minimum=0),
    gr.Checkbox(label="Has Office"),
    gr.Number(label="Toilet Count", precision=0, minimum=0),
    gr.Checkbox(label="Has Swimming Pool"),
    gr.Checkbox(label="Has Fireplace"),
    gr.Number(label="Terrace Surface (m²)", precision=0, minimum=0),
    gr.Dropdown(["A+", "A", "B", "C", "D", "E", "F", "G"], label="EPC Score"),
]

# Mapping between user friendly input names and actual model features
def gradio_wrapper(*args):
    # Map user-friendly inputs to feature names
    subtypes = {
        'CHALET': 0,
        'OTHER_PROPERTY': 1,
        'BUNGALOW': 2,
        'TOWN_HOUSE': 3,
        'HOUSE': 4,
        'COUNTRY_COTTAGE': 5,
        'MIXED_USE_BUILDING': 6,
        'APARTMENT_BLOCK': 7,
        'MANOR_HOUSE': 8,
        'FARMHOUSE': 9,
        'CASTLE': 10,
        'MANSION': 11,
        'VILLA': 12,
        'EXCEPTIONAL_PROPERTY': 13
    }
    conditions = {
        'TO_RESTORE': 0,
        'TO_RENOVATE': 1,
        'TO_BE_DONE_UP': 2,
        'GOOD': 3,
        'JUST_RENOVATED': 4,
        'AS_NEW': 5
    }
    kitchen_types = {
        'NOT_INSTALLED' : 0,
        'SEMI_EQUIPPED' : 1,
        'USA_UNINSTALLED' : 2,
        'USA_SEMI_EQUIPPED' : 3,
        'INSTALLED' : 4,
        'USA_INSTALLED' : 5,
        'HYPER_EQUIPPED' : 6,
        'USA_HYPER_EQUIPPED' : 7
    }
    
    # Convert friendly inputs to the format expected by the model
    subtype_val = subtypes.get(args[0], 0)
    condition_val = conditions.get(args[6], 0)
    kitchen_val = kitchen_types.get(args[12], 0)
    postcode_val = postal_code_prices.get(args[3], 0)
    
    input_values = [
        args[4],              # habitableSurface
        args[16],             # toiletCount
        args[2],
        postcode_val,         # postCode
        args[1],              # bedroomCount
        subtype_val,          # subtype
        kitchen_val,          # kitchenType
        condition_val,        # buildingCondition
        args[13],             # landSurface
        args[15],             # hasOffice
        args[17],             # hasSwimmingPool
        args[20],             # epc_kwh
        args[7],              # facedeCount
        args[19],             # terraceSurface
        args[14],             # parkingCountOutdoor
        args[18],             # hasFireplace
        args[5],              # hasDressingRoom
        args[10],              # hasPhotovoltaicPanels
        args[9],              # hasHeatPump
        args[11],             # hasThermicPanels
        args[7],              # buildingConstructionYear
    ]
    
    # Create dictionary with feature names as keys
    input_dict = dict(zip(FEATURES, input_values))
    print(input_dict)
    return predict(input_dict)

# Create the interface
interface = gr.Interface(
    fn=gradio_wrapper,
    title="Real Estate Price Prediction in Belgium",
    inputs=inputs,
    outputs=gr.Number(label="Price Prediction (€)")
)

if __name__ == "__main__":
    interface.launch(share=False, show_api=False)