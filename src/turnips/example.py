#!/usr/bin/env python3

from turnips import archipelago

EXAMPLE_DATA = '''
{
    "islands": {
        "trimex": {
            "previous_week": "bump",
            "timeline": {
                "Sunday_AM":   110,
                "Monday_AM":   58,
                "Monday_PM":   null,
                "Tuesday_AM":  null,
                "Tuesday_PM":  null,
                "Wednesday_AM":null,
                "Wednesday_PM":null,
                "Thursday_AM": null,
                "Thursday_PM": null,
                "Friday_AM":   null,
                "Friday_PM":   null,
                "Saturday_AM": null,
                "Saturday_PM": null
            }
        },
        "Aduien": {
            "previous_week": "bump",
            "timeline": {
                "Sunday_AM":   91,
                "Monday_AM":   113,
                "Monday_PM":   106,
                "Tuesday_AM":  98,
                "Tuesday_PM":  null,
                "Wednesday_AM":null,
                "Wednesday_PM":null,
                "Thursday_AM": null,
                "Thursday_PM": null,
                "Friday_AM":   null,
                "Friday_PM":   null,
                "Saturday_AM": null,
                "Saturday_PM": null
            }
        },
        "Torishima": {
            "previous_week": "spike",
            "timeline": {
                "Sunday_AM":   106,
                "Monday_AM":   105,
                "Monday_PM":   134,
                "Tuesday_AM":  121,
                "Tuesday_PM":  null,
                "Wednesday_AM":null,
                "Wednesday_PM":null,
                "Thursday_AM": null,
                "Thursday_PM": null,
                "Friday_AM":   null,
                "Friday_PM":   null,
                "Saturday_AM": null,
                "Saturday_PM": null
            }
        },
        "File Isle": {
            "previous_week": "bump",
            "timeline": {
                "Sunday_AM":   96,
                "Monday_AM":   83,
                "Monday_PM":   79,
                "Tuesday_AM":  null,
                "Tuesday_PM":  null,
                "Wednesday_AM":null,
                "Wednesday_PM":null,
                "Thursday_AM": null,
                "Thursday_PM": null,
                "Friday_AM":   null,
                "Friday_PM":   null,
                "Saturday_AM": null,
                "Saturday_PM": null
            }
        },
        "Kibo": {
            "previous_week": "bump",
            "timeline": {
                "Sunday_AM":   93,
                "Monday_AM":   73,
                "Monday_PM":   69,
                "Tuesday_AM":  65,
                "Tuesday_PM":  null,
                "Wednesday_AM":null,
                "Wednesday_PM":null,
                "Thursday_AM": null,
                "Thursday_PM": null,
                "Friday_AM":   null,
                "Friday_PM":   null,
                "Saturday_AM": null,
                "Saturday_PM": null
            }
        },
        "Calidris": {
            "previous_week": "bump",
            "timeline": {
                "Sunday_AM":   99,
                "Monday_AM":   98,
                "Monday_PM":   93,
                "Tuesday_AM":  null,
                "Tuesday_PM":  null,
                "Wednesday_AM":null,
                "Wednesday_PM":null,
                "Thursday_AM": null,
                "Thursday_PM": null,
                "Friday_AM":   null,
                "Friday_PM":   null,
                "Saturday_AM": null,
                "Saturday_PM": null
            }
        },
        "Harvest": {
            "previous_week": "bump",
            "timeline": {
                "Sunday_AM":   110,
                "Monday_AM":   81,
                "Monday_PM":   77,
                "Tuesday_AM":  73,
                "Tuesday_PM":  68,
                "Wednesday_AM":null,
                "Wednesday_PM":null,
                "Thursday_AM": null,
                "Thursday_PM": null,
                "Friday_AM":   null,
                "Friday_PM":   null,
                "Saturday_AM": null,
                "Saturday_PM": null
            }
        },
        "Wolfshire": {
            "initial_week": true,
            "timeline": {
                "Sunday_AM":   107,
                "Monday_AM":   95,
                "Monday_PM":   92,
                "Tuesday_AM":  88,
                "Tuesday_PM":  null,
                "Wednesday_AM":null,
                "Wednesday_PM":null,
                "Thursday_AM": null,
                "Thursday_PM": null,
                "Friday_AM":   null,
                "Friday_PM":   null,
                "Saturday_AM": null,
                "Saturday_PM": null
            }
        },
        "covid19": {
            "initial_week": true,
            "timeline": {
                "Sunday_AM":   97,
                "Monday_AM":   null,
                "Monday_PM":   null,
                "Tuesday_AM":  70,
                "Tuesday_PM":  null,
                "Wednesday_AM":null,
                "Wednesday_PM":null,
                "Thursday_AM": null,
                "Thursday_PM": null,
                "Friday_AM":   null,
                "Friday_PM":   null,
                "Saturday_AM": null,
                "Saturday_PM": null
            }
        }
    }
}
'''

def main() -> None:
    # See also `turnips.py --plot sample.json`
    islands = archipelago.Archipelago.load_json(EXAMPLE_DATA)
    islands.plot()


if __name__ == '__main__':
    main()
