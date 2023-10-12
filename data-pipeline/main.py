from get_data import ImageDownloader

class DataOrchestror():
    def __init__(self) -> None:
        self.teams = [
    "América-MG", "Athletico-PR", "Atlético-MG", "Bahia", "Botafogo",
    "Corinthians", "Coritiba", "Cruzeiro", "Cuiabá", "Flamengo",
    "Fluminense", "Fortaleza", "Goiás", "Grêmio", "Internacional",
    "Palmeiras", "Red Bull Bragantino", "Santos", "São Paulo", "Vasco"
]

    def prepare_data(self):
        ImageDownloader().download_team_images(self.teams)
        

def main():
    orchestror = DataOrchestror()
    orchestror.prepare_data()

if __name__ == '__main__':
    main()