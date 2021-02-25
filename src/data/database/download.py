"""Download and extract NINAPRO databases 1 to 9."""
import os
from copy import deepcopy
import time
import re
import shutil
import zipfile
from pathlib import Path
import requests
from base64 import b64decode, b64encode
import yaml
from bs4 import BeautifulSoup
from starlette.status import HTTP_401_UNAUTHORIZED
from fastapi import HTTPException
from tqdm import tqdm
import math


DATA_FORM_TEMPLATE = {
    'name': '',
    'pass': '',
    'form_id': 'user_login_block'
}
DEFAULT_HEADERS = {
    'Connection': 'keep-alive',
    'Pragma': 'no-cache',
    'Cache-Control': 'no-cache',
    'Upgrade-Insecure-Requests': '1',
    'Origin': 'http://ninapro.hevs.ch',
    'Content-Type': 'application/x-www-form-urlencoded',
    'Referer': 'http://ninapro.hevs.ch/',
}
NO_HREF_CLASSES = ['views-field', 'views-field-title']
TITLE_FIELD_CLASSES = ["views-field-field-db9-index", "views-field-title"]
FILE_CLASS_REGEX = re.compile('.*?file.*|.*CalibratedKinematicData.*', re.I)
CWD = str(Path(os.path.abspath(__file__)).parent)
PATHS = [CWD + "/db{}".format(db + 1) for db in range(9)]


class NinaWeb:
    def __init__(self, config_file="config.yaml"):
        with open(config_file, "r") as stream:
            self.config = yaml.safe_load(stream)
        self._login_cookies = None
        self._download_links = {}

    def login(self):
        user = b64decode(self.config['nina_user'].encode('utf-8'))
        password = b64decode(self.config['nina_password'].encode('utf-8'))
        data = deepcopy(DATA_FORM_TEMPLATE)
        data['name'] = user.decode('utf-8')
        data['pass'] = password.decode('utf-8')
        res = requests.post(
            self.config['base_url'] + '/' + self.config['login_endpoint'],
            data=data,
            headers=DEFAULT_HEADERS,
            allow_redirects=False)
        if not res.ok:
            raise HTTPException(
                status_code=res.status_code,
                detail="Login was not successfull."
            )
        for cookie_name in res.cookies.keys():
            if 'SESS' in cookie_name:
                self._login_cookies = res.cookies
        if not self._login_cookies:
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="No session cookie was retrieved from login."
            )

    def logout(self):
        res = requests.get(
            self.config['base_url'] + '/' + self.config['logout_endpoint'],
            headers=DEFAULT_HEADERS,
            cookies=self._login_cookies
        )
        if not res.ok:
            raise HTTPException(
                status_code=res.status_code,
                detail="Logout was not successfull."
            )
        self._login_cookies = None

    def _get_database_links(self, database):
        database = int(database)
        if database in self._download_links.keys():
            return self._download_links[database]
        url = self.config['base_url'] + \
            self.config['database_endpoints'][database]
        res = requests.get(
            url,
            cookies=self._login_cookies,
            headers=DEFAULT_HEADERS
        )
        if not res.ok:
            raise HTTPException(
                status_code=res.status_code,
                detail="Error when getting url {}".format(url)
            )
        soup = BeautifulSoup(res.content, 'html.parser')
        table = soup.find('table', {'class', 'views-table'}).find('tbody')
        database_links = {}
        for tr in table.find_all('tr'):
            tr_text = tr.find('td', {'class': TITLE_FIELD_CLASSES}).text
            tr_file = tr.find('td', {'class': FILE_CLASS_REGEX}).text
            tr_file = tr_file.replace(' ', '').replace('\n', '')
            subject = int(re.search(r'\d+', tr_text).group())
            files = []
            for a in tr.find_all('a', href=True):
                if set(a.parent['class']).intersection(NO_HREF_CLASSES):
                    continue
                link = a['href']
                if 'http' not in link:
                    link = self.config['base_url'] + link
                if link.endswith('.zip'):
                    filename = link.split("/")[-1]
                    zipped = True
                elif link.lower().endswith('.mat'):
                    filename = link.split("/")[-1]
                    zipped = False
                else:
                    filename = tr_file
                    zipped = filename.endswith('.zip')
                files.append(
                    {
                         'file_name': filename,
                         'zipped': zipped,
                         'link': link
                    }
                )
            database_links[subject] = files
        self._download_links[database] = database_links
        return database_links

    def _get_dataset_link(self, database, dataset):
        return self._get_database_links(database)[dataset]

    def download_dataset(self, database, dataset, force=False):
        """Download subjects files.

        Parameters:
            url (:obj:`str`): url for getting the files links
            path (:obj:`str`): path to the folder where to download and extract the files

        """
        start_time = time.time()
        files = self._get_dataset_link(database, dataset)
        subj_dir = Path('{}/s{}'.format(PATHS[database - 1], dataset))
        subj_dir.mkdir(parents=True, exist_ok=True)
        has_mat = bool([sfile for sfile in subj_dir.rglob('**/*.*')
                        if str(sfile).lower().endswith('.mat')])
        if has_mat and not force:
            print('File was already dowloaded.')
            return
        for nfile in files:
            localfile = Path(str(subj_dir) + "/" + nfile['file_name'])
            print("Starting download of file {} in folder {}...".format(
                nfile['file_name'], subj_dir)
            )
            with requests.get(nfile['link'], stream=True,
                              cookies=self._login_cookies) as response:
                total_size_in_bytes = int(response.headers.get('content-length', 0))
                block_size = 1024 # 1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)                        
                with open(localfile, "wb") as mfile:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        mfile.write(data)
            progress_bar.close()
            print('File {} was downloaded in {} seconds.'.format(
                nfile['file_name'], time.time() - start_time) 
            )
            if nfile["zipped"]:
                extract_zips(str(localfile), str(subj_dir))

    def download_database(self, database, force=False):
        database_links = self._get_database_links(database).keys()
        for dataset in database_links:
            self.download_dataset(database, dataset, force=True)


def extract_zips(zipfilename, path):
    """Extract a dataset zipped file.

    Parameters:
        zipfilename (:obj:`str`): path of the file to extract
        path (:obj:`str`): path to the folder where to extract the files

    """
    zpfile = zipfile.ZipFile(zipfilename, "r")
    files_list = []
    for zfile in zpfile.namelist():
        if "MACOSX" not in zfile and\
        zfile.endswith(".mat")\
        or zfile.endswith(".MAT"):
            files_list.append(zpfile.extract(zfile, path))
    for mats in files_list:
        parts = mats.split("/")
        to_rmpath = "/".join(parts[:-1])
        shutil.move(mats, "{}/{}".format(path, parts[-1]))
    if to_rmpath != path:
        try:
            shutil.rmtree(to_rmpath)
        except Exception as error:
            print ("Error while removing path {}: {}".format(to_rmpath, error))
    try:
        os.remove(zipfilename)
    except Exception as error:
        print("Error while removing zip file {}: {}".format(zipfilename, error))
