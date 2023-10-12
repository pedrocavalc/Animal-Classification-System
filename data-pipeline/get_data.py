from bing_image_downloader import downloader
import os
import shutil
import random

TEAMS = [
    "América-MG", "Athletico-PR", "Atlético-MG", "Bahia", "Botafogo",
    "Corinthians", "Coritiba", "Cruzeiro", "Cuiabá", "Flamengo",
    "Fluminense", "Fortaleza", "Goiás", "Grêmio", "Internacional",
    "Palmeiras", "Red Bull Bragantino", "Santos", "São Paulo", "Vasco"
]

class ImageDownloader:
    @staticmethod
    def download_team_images():
        for team in TEAMS:
            try:
                ImageDownloader.download_images_for_team(team)
            except:
                continue
        ImageDownloader.split_dataset()
        ImageDownloader.delete_folder()

    @staticmethod
    def download_images_for_team(team):
        query = 'Camisa do ' + team
        output_directory = 'Dataset'
        downloader.download(query, limit=100, output_dir=output_directory, adult_filter_off=True, force_replace=False, timeout=1)
    
    @staticmethod
    def split_dataset():
        for team in TEAMS:
            team_dir = os.path.join('Dataset',  'Camisa do ' + team)
            if not os.path.exists(team_dir):
                continue

            images = [f for f in os.listdir(team_dir) if os.path.isfile(os.path.join(team_dir, f))]
            random.shuffle(images)

            train_count = int(0.7 * len(images))
            test_count = int(0.2 * len(images))

            train_images = images[:train_count]
            test_images = images[train_count: train_count + test_count]
            val_images = images[train_count + test_count:]
            for img in train_images:
                train_team_dir = os.path.join('../Dataset', 'train', team)
                if not os.path.exists(train_team_dir):
                    os.makedirs(train_team_dir)
                shutil.move(os.path.join(team_dir, img), os.path.join(train_team_dir, img))

            for img in test_images:
                test_team_dir = os.path.join('../Dataset', 'test', team)
                if not os.path.exists(test_team_dir):
                    os.makedirs(test_team_dir)
                shutil.move(os.path.join(team_dir, img), os.path.join(test_team_dir, img))

            for img in val_images:
                val_team_dir = os.path.join('../Dataset', 'val', team)
                if not os.path.exists(val_team_dir):
                    os.makedirs(val_team_dir)
                shutil.move(os.path.join(team_dir, img), os.path.join(val_team_dir, img))




    @staticmethod
    def delete_folder():
        shutil.rmtree('Dataset/')
    


if __name__ == "__main__":
    ImageDownloader.download_team_images()
