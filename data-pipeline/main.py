from get_data import ImageDownloader

class DataOrchestror():
    def __init__(self) -> None:
        pass

    def prepare_data(self):
        ImageDownloader().download_team_images()
    
        

def main():
    orchestror = DataOrchestror()
    orchestror.prepare_data()

if __name__ == '__main__':
    main()