# About: Search & Scrape GitHub repositories
"""Search on github for keyword(s) with optional parameters to refine your search and get all results in a CSV file.
Doc::

    pip install -e .
    $utilmy = "yourfolder/myutil/utilmy/
    python  $utilmy/webscraper/cli_github_gist_search.py  run



"""
import requests,csv, os
from bs4 import BeautifulSoup as bs


def run(url= "https://gist.github.com/search?p={}&q=pyspark+UDF"  , logs=True, download=True, dwnldDir="./zdown_github/"):
    #logs = True
    #download = True
    #dwnldDir = os.getcwd()
    try:
        os.mkdir(os.path.join(dwnldDir,  "github_files"))
    except:
        pass
    csvFilePath = os.path.join(dwnldDir, "GitHub_files.csv")
    with open(csvFilePath, 'w') as f:
        w = csv.writer(f)
        w.writerow(["File Name", "File Url", "File Download Url"])

    u = url
    filesList = []
    for i in range(1, 53):
        url = u.format(i)
        if logs:
            print("=="*20)
            print(i, url)    
        data = requests.get(url)
        obj = bs(data.text, 'html.parser')
        files = obj.findAll('a', {'class': "link-overlay"})


        for f in files:
            fName = f.find('span').find('strong').text
            fUrl = f['href']
            data = requests.get(fUrl)
            obj = bs(data.text, 'html.parser')
            fileDir = os.path.join(dwnldDir, "github_files", fName.split('.')[0])
            fileDwnldUrl = "https://gist.github.com" + obj.findAll('a', {'data-ga-click': "Gist, download zip, location:gist overview"})[0]['href']
            if download:
                data = requests.get(fileDwnldUrl)
                import zipfile
                from io import BytesIO
                zipfile = zipfile.ZipFile(BytesIO(data.content))
                zipfile.extractall(fileDir)
            with open(csvFilePath, 'a', newline='') as f:
                w = csv.writer(f)
                w.writerow([fName, fUrl, fileDwnldUrl])
            if logs:
                print("\t", fName)
                print("\t", "\t", fUrl)
                print("\t", "\t", fileDwnldUrl)
        if logs:
            print("=="*20)
        else:
            print("InProgress...")
    print("Done")



if __name__ == "__main__":
    import fire 
    fire.Fire()
    #### python = utilmy$/webscraper/cli_github_gist_search.py  run
