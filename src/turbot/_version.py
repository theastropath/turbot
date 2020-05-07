import dunamai as _dunamai

# local development version
_version_from_git = _dunamai.get_version(
    "turbot", first_choice=_dunamai.Version.from_any_vcs
).serialize()

if _version_from_git != "0.0.0":
    __version__ = _version_from_git
else:
    # published release version
    __version__ = _dunamai.get_version(
        "turbot", third_choice=_dunamai.Version.from_any_vcs
    ).serialize()
