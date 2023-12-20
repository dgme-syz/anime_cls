@echo off
set DOWNLOAD_URL=https://public.sn.files.1drv.com/y4m5Xs0Z1GTUm8-ixMUOgx-rJ6nT28xfE814g6uey2iJ_fxyfG1FVyaA8wxnQsoGsBSnFUnn1WAZBDf4FhN7yw-O-NbT4oli6SuJNVCW3wNorDCxXfQT5ofQCv6l2Efj4WLDp3LH6KP8XZ6BvYFuxdQMftb1HFzje2Qbg6iZwP9iPEO7rD2dQxQvfH9J-Q_2Pf_zHaXGbS9icG2ozrug89rZOZBXC4IHsn4DxYXnTkFUCw

set ZIP_FILE=data.zip
set UNZIP_DIR=./data

echo Downloading file...
curl -o %ZIP_FILE% %DOWNLOAD_URL%

echo Unzipping file...
powershell -Command "Expand-Archive -Path %ZIP_FILE% -DestinationPath %UNZIP_DIR%"

echo Cleanup: Removing zip file...
del %ZIP_FILE%

echo Done!
