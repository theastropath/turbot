# Docker Support

Turbot can be run directly from docker so that you don't have to worry
about installing anything or managing your Python environment.

First build your Turbot docker image.

```shell
docker build -t turbot .
```

You can now run it via `docker run`. You'll want to pass your
configuration into the bot via environment variables. You will also want
to mount a volume for the `db` directory so that your application state
doesn't get blown away whenever you kill and remove the image.

```shell
docker run \
    -e TURBOT_TOKEN="<your-discord-bot-token>" \
    -e TURBOT_CHANNELS="<some;channels;here>" \
    -v "$(pwd)/db":/db \
    --rm
    turbot
```
