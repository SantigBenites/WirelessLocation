# Usage of docker compose

## To start

docker compose up -d

## To stop

docker compose down

## Warining

I recomend u create the directory `/home/sbenites/mongodb` so that docker doesnt write to ur home as root, that can be a pain

Use the following comamnds:

```bash
mkdir -p ./mongodb
chmod -R 700 ./mongodb 
```


## Show collections documents

db.collectionName.find()