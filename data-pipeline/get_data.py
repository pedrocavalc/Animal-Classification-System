from bing_image_downloader import downloader
import os
import shutil

class ImageDownloader:
    @staticmethod
    def download_team_images(teams):
        for team in teams:
            try:
                ImageDownloader.download_images_for_team(team)
            except:
                continue
        ImageDownloader.move_folder()

    @staticmethod
    def download_images_for_team(team):
        query = 'Camisa do ' + team
        output_directory = 'Dataset'
        downloader.download(query, limit=100, output_dir=output_directory, adult_filter_off=True, force_replace=False, timeout=1)
    
    @staticmethod
    def move_folder():
        shutil.move('Dataset', os.path.join('../Dataset'))

if __name__ == "__main__":
    ImageDownloader.download_team_images()
