from setuptools import find_packages, setup

setup(
    name="turbot",
    version="2.0.0",
    author="TheAstropath",
    license="MIT",
    packages=find_packages(),
    long_description=open("README.md", "r").read(),
    entry_points={
        "console_scripts": ["turbot=turbot:main", "migrate=turbot.migrate:main"]
    },
    url="https://github.com/theastropath/turbot",
    include_package_data=True,
    author_email="theastropath@gmail.com",
    zip_safe=False,
)
