"""Fetch official source archive and extract only the four required files locally."""
import urllib.request
import zipfile
from pathlib import Path

URL = 'https://downloads.ctfassets.net/psj0p18eh7z1/3e9OAF7F9ONT4pwJc1luEw/d56af8aabad51bdb9888aad0240bd105/dunnhumby_The-Complete-Journey.zip'
raw = Path('data/raw')
raw.mkdir(parents=True, exist_ok=True)
archive = raw/'complete_journey.zip'
if not archive.exists():
    partial = archive.with_suffix('.partial')
    urllib.request.urlretrieve(URL, partial)
    partial.rename(archive)
with zipfile.ZipFile(archive) as z:
    for name in z.namelist():
        if name.endswith(('transaction_data.csv', 'product.csv', 'hh_demographic.csv', '.pdf')):
            (raw/Path(name).name).write_bytes(z.read(name))
print('Extracted locally. Do not commit or redistribute the raw source files.')
