from sklearn.base import BaseEstimator, TransformerMixin

country_mapping = {
        "Abarth": "Italy",
        "Acura": "Japan",
        "Alfa Romeo": "Italy",
        "Aston Martin": "United Kingdom",
        "Audi": "Germany",
        "BMW": "Germany",
        "Baic": "China",
        "Bentley": "United Kingdom",
        "Bestune": "China",
        "Brilliance": "China",
        "Bugatti": "France",
        "Buick": "United States",
        "Byd": "China",
        "Cadillac": "United States",
        "Canghe": "China",
        "Chana": "China",
        "Changan": "China",
        "Chery": "China",
        "Chevrolet": "United States",
        "Chrysler": "United States",
        "Citroën": "France",
        "Cupra": "Spain",
        "DFSK": "China",
        "Daewoo": "South Korea",
        "Daihatsu": "Japan",
        "Datsun": "Japan",
        "Dodge": "United States",
        "Domy": "China",
        "Dongfeng": "China",
        "Ds": "France",
        "Emgrand": "China",
        "Faw": "China",
        "Fiat": "Italy",
        "Ford": "United States",
        "Forthing": "China",
        "Foton": "China",
        "GAC": "China",
        "Gaz": "Russia",
        "Geely": "China",
        "Gmc": "United States",
        "Great Wall": "China",
        "Hafei": "China",
        "Haima": "China",
        "Haval": "China",
        "Hawtai": "China",
        "Honda": "Japan",
        "Hummer": "United States",
        "Hyundai": "South Korea",
        "Infiniti": "Japan",
        "Isuzu": "Japan",
        "Jac": "China",
        "Jaguar": "United Kingdom",
        "Jeep": "United States",
        "Jetour": "China",
        "Jonway": "China",
        "Kaiyi": "China",
        "Karry": "China",
        "Kenbo": "China",
        "Keyton": "China",
        "Kia": "South Korea",
        "Lada": "Russia",
        "Lancia": "Italy",
        "Land Rover": "United Kingdom",
        "Landwind": "China",
        "Lexus": "Japan",
        "Lifan": "China",
        "Lincoln": "United States",
        "Lotus": "United Kingdom",
        "MG": "United Kingdom",
        "Mahindra": "India",
        "Maserati": "Italy",
        "Mazda": "Japan",
        "McLaren": "United Kingdom",
        "Mercedes": "Germany",
        "Mini": "United Kingdom",
        "Mitsubishi": "Japan",
        "Nissan": "Japan",
        "Opel": "Germany",
        "Perodua": "Malaysia",
        "Peugeot": "France",
        "Porsche": "Germany",
        "Proton": "Malaysia",
        "Renault": "France",
        "Saipa": "Iran",
        "Seat": "Spain",
        "Senova": "China",
        "Skoda": "Czech Republic",
        "Smart": "Germany",
        "Sokon": "China",
        "Soueast": "China",
        "Speranza": "China",
        "Ssang Yong": "South Korea",
        "Subaru": "Japan",
        "Suzuki": "Japan",
        "Tata": "India",
        "Tesla": "United States",
        "Toyota": "Japan",
        "Volkswagen": "Germany",
        "Volvo": "Sweden",
        "Zeekr": "China",
        "Zotye": "China"
    }

color_mapping = {
    'Mocha': 'Brown',
    'Dark green': 'Green',
    'Olive': 'Green',
    'Champagne': 'Gold',
    'Yellow': 'Gold',
    'Eggplant': 'Purple'
}

class CarUsedFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["Relative Ad Date"] = (
            X["Date Displayed"].max() - X["Date Displayed"]
        ).dt.days

        X["Car Age"] = X["Year"].max() - X["Year"]

        X["Mileage Per Year"] = X["Mileage"] / (X["Car Age"] + 1)

        X["Country_of_Origin"] = X["Make"].map(country_mapping)

        X['Color'] = X["Color"].replace(color_mapping)

        X = X.drop(columns=["Date Displayed"])

        return X

    def get_feature_names_out(self, input_features=None):

        features = list(input_features)
        features.remove("Date Displayed")
        features.extend([
            "Relative Ad Date",
            "Car Age",
            "Mileage Per Year",
            "Country_of_Origin"
        ])

        return features    