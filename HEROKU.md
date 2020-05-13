# Heroku Support

The `bin` directory contains scripts for use with [Heroku][heroku].
Additionally, the files `Procfile`, `requirements.txt`, `runtime.txt`,
and `app.json` all exist to support Heroku.

There's a one-click to deploy button on the main [README](README.md) of this
repository. If you'd rather go through the process manually, read further.

## HOWTO: Manually Deploy Turbot to Heroku

1. Create a Heroku account if you don't already have one.
   Free tier accounts are available.
2. Create a new app on your account.
3. Install the [Heroku CLI][heroku-cli] on your computer.
4. If you haven't already, clone this repository with [git][git].
5. Make sure you have the authorization token for your bot from Discord.
   Please see [Discord's own documentation on creating a bot][discord-bot]
   if you have not already created the bot on Discord.
6. Deploy Turbot:

    ```shell
    cd turbot
    heroku login
    heroku git:remote -a <your-heroku-app-name>
    heroku config:set TURBOT_TOKEN=<your-discord-bot-token>
    heroku config:set TURBOT_CHANNELS=<your;authorized;channel;names;>
    heroku buildpacks:clear
    heroku buildpacks:add heroku/python
    git push heroku master
    ```

7. Go to the resources tab in your Heroku app, you should see a worker there.
   Click the edit button and turn the worker on.

At this point your worker should be running. If you encounter issues, check your
Heroku app logs. You can stop the worker from within the settings tab where you
previously enabled it, or directly via the CLI:

```shell
heroku ps:scale worker=0
```

Then to restart it:

```shell
heroku ps:scale worker=1
```

[heroku]:       https://heroku.com/
[heroku-cli]:   https://devcenter.heroku.com/articles/heroku-cli
[git]:          https://git-scm.com/
[discord-bot]:  https://discord.com/developers/docs/intro
