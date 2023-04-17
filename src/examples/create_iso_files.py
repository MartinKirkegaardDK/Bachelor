from utils.data_transformation import load_iso_codes

def run():
    continents = ["africa","americas","asia","europe","oceania"]
    for elm in continents:
        t = load_iso_codes(elm)
        with open(f'data/iso_codes/{elm}.txt', 'w') as f:
            for line in t:
                f.write(f"{line}\n")