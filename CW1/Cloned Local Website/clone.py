import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from flask import Flask, send_file

app = Flask(__name__)


@app.route("/")
def serve_website():
    return send_file("index.html")


def download_website(URL):
    """
    function: download_website

    arg:
        -url (string): URL of the target website to clone (e.g www.facebook.com)

    returns:
        true/false if download was a sucess
    
    example:
        python3 download_website(www.facebook.com)

    """
    # downloads all avaibale assets and hosts them
    try:
        #checks website response
        response = requests.get(url)

        #if active
        if response.status_code == 200:
            red_dir = "downloaded_resources"
            ensure_directory(red_dir)
            soup = BeautifulSoup(response.content, "html.parser")
            #attempts to find all attached assets, including hyperlinks, images and JS scripts
            for tag in soup.find_all(["script", "link", "img"]):
                try:
                    # download src tag
                    if tag.get("src"):
                        src = tag["src"]
                        if not src.startswith("http"):
                            src = urljoin(url, src)
                        local_src = os.path.join(red_dir, os.path.basename(src))
                        tag["src"] = local_src
                        with open(local_src, "wb") as f:
                            f.write(requests.get(src).content)
                    # download href tag
                    elif tag.get("href"):
                        href = tag["href"]
                        if not href.startswith("http"):
                            href = urljoin(url, href)
                        local_href = os.path.join(red_dir, os.path.basename(href))
                        tag["href"] = local_href
                        with open(local_href, "wb") as f:
                            f.write(requests.get(href).content)

                # if it cant be downloaded it skips it (tends to tirgger on images or payment sections)
                except:
                    print(f"Error downloading:{tag}")
            with open("index.html", "w", encoding="utf-8") as f:
                f.write(str(soup))

            return True
        else:
            print(
                "Failed to download the website. Code:",
                response.status_code,
            )
            return False
    except Exception as e:
        print("An error occurred while downloading the website:", e)
        return False

#checks directory existance
def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == "__main__":
    # hosts cloned website on flask local server. Same computer IP port 8000
    website_url = input("Enter the URL of the website to download and host locally: ")

    #attempts to host the website
    if download_website(website_url):
        print("Website downloaded successfully.")
        app.run(host="0.0.0.0", port=8000)
    else:
        print("Failed to download the website.")
