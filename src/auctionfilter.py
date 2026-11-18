from difflib import SequenceMatcher
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"

# Common surnames where (first_initial + last_name) could collide across different players
COMMON_SURNAMES = {
    "sharma", "singh", "kumar", "yadav", "khan", "patel", "verma", "reddy",
    "rao", "nair", "das", "roy", "ali", "gill", "pandya", "iyer", "dubey",
    "tiwari", "mishra", "pandey", "gupta", "joshi", "shah", "shetty",
    "kulkarni", "chahal", "brar", "sandhu", "sidhu", "dhillon", "bawa",
    "rawat", "chauhan", "solanki", "jadeja", "chawla", "arora", "malhotra",
    "kapoor", "khanna", "thakur", "chaudhary", "choudhary", "ansari",
    "ahmed", "hussain", "hasan", "rahman", "alam", "uddin", "dar", "shukla",
}

# 1. All players retained or traded into IPL 2027 squads (Nov 2026 retention lists)
RETAINED_IPL_2027 = {
    # RR (19)
    "Yashasvi Jaiswal", "Vaibhav Sooryavanshi", "Riyan Parag", "Dhruv Jurel",
    "Ravindra Jadeja", "Shubham Dubey", "Aman Rao Perala", "Brijesh Sharma",
    "Mohammed Shami", "Mohd Shami", "Ravi Bishnoi", "Sandeep Sharma",
    "Manimaran Siddharth", "M Siddharth", "Sushant Mishra", "Yash Raj Punja",
    "Jofra Archer", "Nandre Burger", "Donovan Ferreira", "Dasun Shanaka",
    "Lhuan-dre Pretorius", "L Pretorius",
    # CSK (19)
    "Ruturaj Gaikwad", "Sanju Samson", "Shivam Dube", "Ayush Mhatre",
    "Dewald Brevis", "Liam Livingstone", "Jamie Overton", "Jack Edwards",
    "Noor Ahmad", "Akeal Hosein", "Rahul Chahar", "Mukesh Choudhary",
    "Anshul Kamboj", "Gurjapneet Singh", "Kartik Sharma", "Ramakrishna Ghosh",
    "Sarfaraz Khan", "Shreyas Gopal", "Urvil Patel",
    # DC (16)
    "KL Rahul", "Rishabh Pant", "Axar Patel", "Tristan Stubbs", "David Miller",
    "Ashutosh Sharma", "Sameer Rizvi", "Prithvi Shaw", "Mukesh Kumar",
    "Lungisani Ngidi", "Lungi Ngidi", "Kyle Jamieson", "Vipraj Nigam",
    "Zeeshan Ansari", "Auqib Dar", "Auqib Nabi", "Madhav Tiwari", "Tripurana Vijay",
    # GT (22)
    "Shubman Gill", "Jos Buttler", "Sai Sudharsan", "Glenn Phillips",
    "Rahul Tewatia", "Washington Sundar", "Rashid Khan", "Kagiso Rabada",
    "Mohammed Siraj", "Mohammad Siraj", "Prasidh Krishna", "Harshal Patel",
    "Jason Holder", "R Sai Kishore", "Sai Kishore", "Tom Banton", "Anuj Rawat",
    "Ashok Sharma", "Gurnoor Singh Brar", "Gurnoor Brar", "Jayant Yadav",
    "Kumar Kushagra", "Manav Suthar", "Mohd Arshad Khan", "Arshad Khan",
    "Nishant Sindhu",
    # KKR (17)
    "Rinku Singh", "Sunil Narine", "Varun Chakaravarthy", "Varun Chakravarthy",
    "CV Varun", "Harshit Rana", "Cameron Green", "Rovman Powell", "Finn Allen",
    "Tim Seifert", "Angkrish Raghuvanshi", "Ramandeep Singh", "Vaibhav Arora",
    "Anukul Roy", "Kartik Tyagi", "Karthik Tyagi", "Manish Pandey",
    "Prashant Solanki", "Sarthak Ranjan", "Saurabh Dubey",
    # LSG (18)
    "Nicholas Pooran", "Mitchell Marsh", "Josh Inglis", "Sam Curran",
    "Ayush Badoni", "Kuldeep Yadav", "Mayank Yadav", "Mohsin Khan",
    "Digvesh Rathi", "Mukul Choudhary", "Arshin Kulkarni", "Abdul Samad",
    "Aiden Markram", "Akash Singh", "Akshat Raghuwanshi", "Arjun Tendulkar",
    "Prince Yadav", "Shahbaz Ahmed",
    # MI (16)
    "Suryakumar Yadav", "Rohit Sharma", "Tilak Varma", "Jasprit Bumrah",
    "Arshdeep Singh", "Ashwani Kumar", "Mitchell Santner", "Ryan Rickelton",
    "Sherfane Rutherford", "Naman Dhir", "Shardul Thakur", "Corbin Bosch",
    "Raghu Sharma", "Quinton de Kock", "Will Jacks", "Raj Angad Bawa",
    # PBKS (19)
    "Shreyas Iyer", "Hardik Pandya", "Priyansh Arya", "Prabhsimran Singh",
    "Shashank Singh", "Marcus Stoinis", "Marco Jansen", "Azmatullah Omarzai",
    "Cooper Connolly", "Yuzvendra Chahal", "Harpreet Brar", "Xavier Bartlett",
    "Ben Dwarshuis", "Vyshak Vijaykumar", "Yash Thakur", "Suryansh Shedge",
    "Vishnu Vinod", "Harnoor Pannu", "Musheer Khan",
    # RCB (20)
    "Virat Kohli", "Rajat Patidar", "Phil Salt", "Tim David", "Jitesh Sharma",
    "Venkatesh Iyer", "Krunal Pandya", "Romario Shepherd", "Josh Hazlewood",
    "Bhuvneshwar Kumar", "Yash Dayal", "Suyash Sharma", "Nuwan Thushara",
    "Jacob Duffy", "Rasikh Dar", "Rasikh Salam", "Devdutt Padikkal",
    "Jordan Cox", "Kanishk Chouhan", "Vicky Ostwal", "Abhinandan Singh",
    # SRH (18)
    "Heinrich Klaasen", "Abhishek Sharma", "Ishan Kishan", "Nitish Kumar Reddy",
    "Pat Cummins", "Nathan Ellis", "Khaleel Ahmed", "T Natarajan",
    "Brydon Carse", "Eshan Malinga", "Aniket Verma", "Harsh Dubey",
    "Smaran Ravichandran", "Krains Fuletra", "Praful Hinge", "Sakib Hussain",
    "Salil Arora", "Shivang Kumar",
}

# 2. All retired / coaching-transition players (2022-2026)
RETIRED_PLAYERS = {
    # New Zealand
    "Tim Southee", "TG Southee", "T Southee",
    "Kane Williamson", "KS Williamson",
    "Martin Guptill", "MJ Guptill",
    "Colin Munro", "C Munro",
    "Neil Wagner", "N Wagner",
    # India
    "MS Dhoni",
    "Ajinkya Rahane", "AM Rahane",
    "Ravichandran Ashwin", "R Ashwin",
    "Dinesh Karthik", "KD Karthik",
    "Shikhar Dhawan", "S Dhawan",
    "Wriddhiman Saha", "WP Saha",
    "Ambati Rayudu", "AT Rayudu",
    "Piyush Chawla", "PP Chawla",
    "Amit Mishra", "A Mishra",
    "Vijay Shankar", "V Shankar",
    "KS Bharat", "Srikar Bharat",
    "Cheteshwar Pujara", "CA Pujara",
    "Kedar Jadhav", "KM Jadhav",
    "Umesh Yadav", "UT Yadav",
    "Varun Aaron", "VR Aaron",
    "Siddarth Kaul", "S Kaul",
    "Murali Vijay", "Robin Uthappa", "Suresh Raina", "Yuvraj Singh",
    "Harbhajan Singh", "Hanuma Vihari", "Dhawal Kulkarni", "Barinder Sran",
    # Australia
    "Glenn Maxwell", "Glen Maxwell", "GJ Maxwell", "G Maxwell",
    "David Warner", "DA Warner",
    "Matthew Wade", "MS Wade",
    "Aaron Finch", "AJ Finch",
    "Kane Richardson", "KW Richardson",
    "Usman Khawaja", "UT Khawaja",
    # England
    "Moeen Ali", "MM Ali", "M Ali",
    "Ben Stokes", "BA Stokes",
    "Mark Wood", "MA Wood",
    "Dawid Malan", "DJ Malan",
    "Chris Jordan", "CJ Jordan",
    "Alex Hales", "AD Hales",
    "Eoin Morgan", "James Anderson", "Stuart Broad",
    # West Indies
    "Andre Russell", "AD Russell",
    "Kieron Pollard", "KA Pollard",
    "Dwayne Bravo", "DJ Bravo",
    "Shannon Gabriel",
    # South Africa & Zimbabwe & Associates
    "Rassie van der Dussen", "HE van der Dussen",
    "Dean Elgar", "D Elgar",
    "Colin Ingram", "CA Ingram",
    "David Wiese", "D Wiese",
    "Navneet Dhaliwal",
    # Bangladesh & Sri Lanka
    "Shakib Al Hasan",
    "Tamim Iqbal",
    "Mahmudullah",
    "Mushfiqur Rahim",
    "Rubel Hossain",
    "Imrul Kayes",
    "Dimuth Karunaratne",
}

# 3. Active players who opted out, withdrew, or are sidelined for IPL 2027
OPTED_OUT_IPL_2027 = {
    "Travis Head", "TM Head",
    "Mitchell Starc", "MA Starc",
    "Faf du Plessis", "F du Plessis",
    "Jacob Bethell", "JG Bethell",
    "Joe Root", "JE Root",
    "Chris Woakes", "CR Woakes",
}

# 4. Fallback Pakistani player names (in addition to automatic ID detection)
PAKISTAN_NAMES_FALLBACK = {
    "Mohammad Rizwan", "Babar Azam", "Shaheen Afridi", "Shaheen Shah Afridi",
    "Haris Rauf", "Naseem Shah", "Saim Ayub", "Fakhar Zaman", "Shadab Khan",
    "Mohammad Nawaz", "Iftikhar Ahmed", "Imad Wasim", "Mohammad Amir",
    "Abrar Ahmed", "Salman Agha", "Salman Ali Agha", "Usman Khan",
    "Mohammad Haris", "Hasan Ali", "Abbas Afridi", "Zaman Khan",
    "Mohammad Wasim", "Mohammad Wasim Jr", "Khushdil Shah", "Asif Ali",
    "Haider Ali", "Shan Masood", "Azam Khan", "Sufiyan Muqeem",
    "Sarfaraz Ahmed",
}


def norm_name(name: str) -> str:
    s = re.sub(r"[^a-z0-9\s]", " ", str(name).lower())
    return " ".join(s.split())


def build_lookup_keys(name_set: set[str]) -> tuple[set[str], set[tuple[str, str]]]:
    exact = set()
    init_last = set()
    for raw in name_set:
        n = norm_name(raw)
        if not n:
            continue
        exact.add(n)
        parts = n.split()
        if len(parts) >= 2:
            last = parts[-1]
            if last not in COMMON_SURNAMES and len(last) >= 4:
                init_last.add((parts[0][0], last))
    return exact, init_last


def matches_group(name: str, exact_set: set[str], init_last_set: set[tuple[str, str]]) -> bool:
    n = norm_name(name)
    if n in exact_set:
        return True

    parts = n.split()
    if len(parts) >= 2:
        first_init, last = parts[0][0], parts[-1]
        if last not in COMMON_SURNAMES and (first_init, last) in init_last_set:
            return True
        if len(last) >= 6:
            for ref_init, ref_last in init_last_set:
                if ref_init == first_init and SequenceMatcher(None, last, ref_last).ratio() >= 0.88:
                    return True

    for ref in exact_set:
        if SequenceMatcher(None, n, ref).ratio() >= 0.90:
            return True

    return False


def get_pakistan_player_ids() -> set:
    pak_ids = set()
    teams_path = DATA / "teams.csv"
    map_path = DATA / "competition_team_player_mapping.csv"

    if teams_path.exists() and map_path.exists():
        teams = pd.read_csv(teams_path)
        mapping = pd.read_csv(map_path)

        team_id_col = next((c for c in teams.columns if "team" in c and "id" in c), None)
        map_team_col = next((c for c in mapping.columns if "team" in c and "id" in c), None)
        map_player_col = next((c for c in mapping.columns if "player" in c and "id" in c), None)

        if team_id_col and map_team_col and map_player_col:
            str_cols = teams.select_dtypes(include="object").columns
            is_pak = pd.Series(False, index=teams.index)
            for col in str_cols:
                is_pak |= teams[col].astype(str).str.strip().str.lower().eq("pakistan")

            pak_team_ids = set(teams.loc[is_pak, team_id_col].dropna())
            pak_ids = set(mapping.loc[mapping[map_team_col].isin(pak_team_ids), map_player_col].dropna())

    return pak_ids


def filter_auction_pool(role: str, pak_ids: set) -> pd.DataFrame:
    df = pd.read_csv(OUT / f"all_{role}_rankings.csv")

    ret_exact, ret_il = build_lookup_keys(RETAINED_IPL_2027)
    retire_exact, retire_il = build_lookup_keys(RETIRED_PLAYERS)
    opt_exact, opt_il = build_lookup_keys(OPTED_OUT_IPL_2027)
    pak_exact, pak_il = build_lookup_keys(PAKISTAN_NAMES_FALLBACK)

    def exclusion_reason(row) -> str:
        pid = row["player_id"]
        name = str(row["player"]).strip()
        if pid in pak_ids or matches_group(name, pak_exact, pak_il):
            return "Pakistan"
        if matches_group(name, ret_exact, ret_il):
            return "Retained/Traded"
        if matches_group(name, retire_exact, retire_il):
            return "Retired"
        if matches_group(name, opt_exact, opt_il):
            return "Opted Out"
        return "AVAILABLE"

    df["status"] = df.apply(exclusion_reason, axis=1)

    removed = df[df["status"] != "AVAILABLE"]
    print(f"\n[{role.upper()} FILTER AUDIT] Total eligible: {len(df)} | Removed: {len(removed)} | Auction Pool: {len(df) - len(removed)}")
    for reason, grp in removed.groupby("status"):
        names = ", ".join(grp["player"].tolist())
        print(f"  - {reason} ({len(grp)}): {names}")

    avail = df[df["status"] == "AVAILABLE"].copy()
    avail["auction_rank"] = range(1, len(avail) + 1)

    cols = [
        "auction_rank", "rank", "player", "balls_hp", "matches_hp",
        "ipl_share_hp", "wpa_hp_shr", "rae_hp_shr", "consistency", "pressure_rating"
    ]
    avail = avail[cols].copy()
    avail["ipl_share_%"] = (avail["ipl_share_hp"] * 100).round(1)
    avail["wpa_hp_100b"] = (avail["wpa_hp_shr"] * 100).round(2)
    avail["rae_hp_100b"] = (avail["rae_hp_shr"] * 100).round(1)
    avail["consistency"] = avail["consistency"].round(3)
    return avail.drop(columns=["ipl_share_hp", "wpa_hp_shr", "rae_hp_shr"])


def main():
    pak_ids = get_pakistan_player_ids()
    bat_avail = filter_auction_pool("batter", pak_ids)
    bowl_avail = filter_auction_pool("bowler", pak_ids)

    bat_avail.to_csv(OUT / "rr_auction_targets_batters.csv", index=False)
    bowl_avail.to_csv(OUT / "rr_auction_targets_bowlers.csv", index=False)

    print("\n=== TOP 15 ACTIVE AUCTION-AVAILABLE CLUTCH BATTERS FOR RR ===")
    print(bat_avail.head(15).to_string(index=False))

    print("\n=== TOP 15 ACTIVE AUCTION-AVAILABLE CLUTCH BOWLERS FOR RR ===")
    print(bowl_avail.head(15).to_string(index=False))


if __name__ == "__main__":
    main()