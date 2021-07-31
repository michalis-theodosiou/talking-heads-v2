import gdown
import shutil
import os

url_root = r'https://docs.google.com/uc?export=download&id='
output_root = 'C:\\Users\\theod\\OneDrive\\Documents\\GitHub\\talkingheads\\MEAD AUDIO\\'

dict = {
    'M003': '1vyV7oBL9qvdA_FRvC5mMU_xNwvxMhivm',
    'M005': '1xTXbLyTSnzy9BHdP9fue_DZzwwGk_2aa',
    'M007': '19quiutoiOQUYxJwkfRdeXB8WXi0gzPPe',
    'M009': '1uhZhONxZvGc3W8oOLnX1EcqBawlgHrmA',
    'M011': '10vyLn2Ck4unShMbEXVKtTZTX527S9xy1',
    'M012': '1AlLEY3bPyNgoLLMYfI3KESNHhd7F9OnG',
    'M013': '1J2ns4DtLs7ohwH-OYx3mg_p7XwwulkMA',
    'M019': '19e3imqFUxcfYI47eOoyXjA5fJOu6krNa',
    'M022': '177Js9WWiBaAXwM9_XLtS6uWZEQ6Lpgu6',
    'M023': '11u9yzMmskOOjbduXMHYOg3rXvQ_Uc7lT',
    'M024': '1BfPM3RQH5Iqe21UBCB222aXjKoehzCSq',
    'M025': '1Xwpgt--o-FyC1AhKVLh6KoI7ioV7Qn4W',
    'M026': '17dFKfM0vGMWztA3lItRJp6rydXZlZQgR',
    'M027': '1k01Wz4qEY5LxPrH40YF9P3oP4bnXWzTc',
    'M028': '1R9yPucDeMldV0pm8MaqSoF7nD7E0hSjM',
    'M029': '1VSyOYtUytJIaT9tU-SfYyiBdl7ZePOiA',
    'M030': '15pdbceiEjumdU76s7zDxB_r7GXkzSlGa',
    'M031': '1-XhN3aUvPDAYt2uVENwF17Ud0pT3W2Gk',
    'M032': '1vIt8JixI8gRkxrzOTiBS7xgOLmg04zKT',
    'M033': '1v8I0icIgdVtz0RH1aU5KJwOCrSRZzjYt',
    'M034': '1JB0Tyn7Kwl3mcpi7nnEc8glTPsLyCkU2',
    'M035': '1o19VqRF1aIJdt4kub95bE1MI_RvUZy8d',
    'M037': '1j_Lih5UZ0JnXMf5synTv9YdcpdAALXcN',
    'M039': '1s6KXcgSxdxCnMh1PPRcB9_GOV_f4Fmj0',
    'M040': '1GT7KVcskw-1Gag3qEt2Y2PUgE52p2a0y',
    'M041': '1KR2RqhLL4ud-9GrTfDP2VkvgamnlQOYc',
    'M042': '1acks6dCogCd_pXBhy2aYFDuqT4UIpC8p',
    'W009': '1ejafqPrZXpIFj8T_D4Gprz26FxFpnSCv',
    'W011': '10vyLn2Ck4unShMbEXVKtTZTX527S9xy1',
    'W014': '1Q_0-f-PdHtgyxaK_ikdOf__VcxqqzoN8',
    'W015': '1IXdrfzKbX5q2fuxkm-cj8COCMfFHlRt0',
    'W016': '1A_JSxZEYQMmEL9qdy_NVYbGVb8SElTZh',
    'W017': '1-lamgaYqnUWwL1-w8Si90wrOGM_nZLlw',
    'W018': '12Z1pS7ArgIvT0tUD6xEM2Y36uXlJRxds',
    'W019': '19e3imqFUxcfYI47eOoyXjA5fJOu6krNa',
    'W021': '1Elk2PhqOMfcxAgxHzHJCqhpvMiHsfRr2',
    'W023': '1ocLJ0Ji4tbmC-Ti44V3bTX9Oxx7dTqPa',
    'W024': '1SYvY1ed5ng9gp_TlSpAnau69NuuP5VAX',
    'W025': '11Mmf3rWaoVMfKjNFKxAUj437xE3hAZuQ',
    'W026': '1_QIlNtaRfqNCxJZUoRxGfLG5b3ZRMisu',
    'W028': '1_tlXWU2ejq_eE1-33p0m4FvdLozQjjmQ',
    'W029': '1_tlXWU2ejq_eE1-33p0m4FvdLozQjjmQ',
    'W033': '1E99anBiX9w5h-byCV01jKmL5qPAJyfKx',
    'W035': '1DMSEZrKs9PBk-SdwUO8ZvKJ9ITRzpX4g',
    'W036': '1P9uJtZRx3AOx6NuOGn28mOy3Sigzzq_4',
    'W037': '1fd8SwnBl49KwigT5PO0VZ8ExzXmukR2Z',
    'W038': '1hFdZ0dxUuMXIoCNpdOiLK6Nd9FWufQGr',
    'W040': '1pAzs-qkGVzaZHv0m9eazCgwYsxpefrYS'
}

assert(len(list(dict.keys())) == len(set(dict.keys())))

for file_no, file_id in dict.items():
    url = url_root + file_id
    output = output_root + str(file_no) + '.tar'

    gdown.download(url, output, quiet=True)
    shutil.unpack_archive(output, 'MEAD AUDIO\\'+str(file_no))
    os.remove(output)