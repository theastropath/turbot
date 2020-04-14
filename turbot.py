import discord
import time
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

tokenFile = open("token.txt",'r')
token = tokenFile.readline().strip() #Strip it, just in case there is whitespace
tokenFile.close()

client = discord.Client()
numToStore = 20
authChannels=["turbot-test","economic-destruction","old-things-for-blathers"]
priceFolder = "prices"
fossilFolder = "fossils"

graphcmdFile = "graphcmd.png"
lastweekFile = "lastweek.png"

twelvehour = mdates.HourLocator(interval=12)
hours = mdates.HourLocator()
hours_fmt = mdates.DateFormatter('%b %d %H:%M')

baseFossilList = ["acanthostega",
               "amber",
               "ammonite",
               "ankylo skull",
               "ankylo torso",
               "ankylo tail",
               "anomalocaris",
               "archaeopteryx",
               "archelon skull",
               "archelon tail",
               "australopith",
               "brachio skull",
               "brachio chest",
               "brachio pelvis",
               "brachio tail",
               "coprolite",
               "deinony torso",
               "deinony tail",
               "dimetrodon skull", 
               "dimetrodon torso",
               "dinosaur track",
               "diplo skull",
               "diplo neck",
               "diplo chest",
               "diplo pelvis",
               "diplo tail", 
               "diplo tail tip",
               "dunkleosteus",
               "eusthenopteron",
               "iguanodon skull",
               "iguanodon torso",
               "iguanodon tail",
               "juramaia",
               "mammoth skull",
               "mammoth torso",
               "megacero skull",
               "megacero torso",
               "megacero tail",
               "left megalo side",
               "right megalo side",
               "myllokunmingia",
               "ophthalmo skull",
               "ophthalmo torso",
               "pachy skull",
               "pachy tail",
               "parasaur skull",
               "parasaur torso",
               "parasaur tail",
               "plesio skull",
               "plesio tail",
               "plesio body",
               "ptera body",
               "left ptera wing",
               "right ptera wing",
               "quetzal torso",
               "left quetzal wing",
               "right quetzal wing",
               "sabertooth skull",
               "sabertooth tail",
               "shark-tooth pattern",
               "spino skull",
               "spino torso",
               "spino tail",
               "stego skull",
               "stego torso",
               "stego tail",
               "t. rex skull",
               "t. rex torso",
               "t. rex tail",
               "tricera skull",
               "tricera torso",
               "tricera tail",
               "trilobite"]

def lookupIdByName(message,name):
    channel = message.channel
    for member in channel.members:
        if name.lower() in str(member).lower():
            return member.id

def lookupUserName(message,name):
    channel = message.channel
    for member in channel.members:
        if name.lower() in str(member).lower():
            return str(member)

def lookupNameById(message,userId):
    channel = message.channel
    for member in channel.members:
        if int(userId) == member.id:
            return str(member)

def loadFossilFileToList(fileName):
    fossils = []

    if os.path.exists(fossilFolder+"/"+fileName):
        f = open(fossilFolder+"/"+fileName,'r')
        for line in f:
            fossils.append(line.strip())
        f.close()
    else:
        fossils = baseFossilList.copy()


    return fossils

def saveFossilListToFile(fileName,fossils):
    f = open(fossilFolder+"/"+fileName,'w')
    for fossil in fossils:
        f.write(fossil+"\n")
    f.close()

def loadPricesToList(fileName):
    prices = []
    try:
        f = open(priceFolder+"/"+fileName,'r')
        for line in f:
            prices.append(line.strip().split(" "))
        f.close()
    except:
        pass

    return prices

def savePriceListToFile(fileName,priceList):
    f = open(priceFolder+"/"+fileName,'w')
    for price in priceList:
        f.write(price[0]+" "+price[1]+" "+price[2]+"\n")
    f.close()


def savePrice(price,priceType,userId):
    print("Saving "+priceType+" price of "+price+" bells for user id "+str(userId))
    fileName = str(userId)+".txt"
    prices = loadPricesToList(fileName)
    newPrice = []
    timestamp = str(time.time())
    newPrice.append(priceType)
    newPrice.append(price)
    newPrice.append(timestamp)
    prices.append(newPrice)

    if len(prices)>numToStore:
        prices = prices[1:]

    savePriceListToFile(fileName,prices)

def getLastSellPrice(userId):
    fileName = str(userId)+".txt"
    prices = loadPricesToList(fileName)
    lastSellPrice = 0
    for price in prices:
        if (price[0]=="sell"):
            lastSellPrice = int(price[1])
    return lastSellPrice

def helloCmd(message):
    print("Sent from "+str(message.author)+", ID is "+str(message.author.id))
    return "Hello!"

def lookupCmd(message):
    print("Lookup")
    response = ""
    splitmsg = message.content.split(" ")
    if len(splitmsg)>1:
        name = " ".join(splitmsg[1:])
        print("Looking up "+name)
        id = lookupIdByName(message,name)
        response = str(id)
    return response

def sellCmd(message):
    print("Log selling price")
    response = ""
    splitmsg = message.content.split(" ")
    lastSellPrice = getLastSellPrice(message.author.id)
    if len(splitmsg)>1:
        price = str(splitmsg[1])
        if price.isnumeric():
            savePrice(price,"sell",message.author.id)
            response = "Logged selling price of "+str(price)+" for user "+str(message.author)
            if lastSellPrice!=0:
                priceInt = int(price)
                if priceInt > lastSellPrice:
                    response+=" (Higher than last selling price of "+str(lastSellPrice)+" bells)"
                elif priceInt < lastSellPrice:
                    response+=" (Lower than last selling price of "+str(lastSellPrice)+" bells)"
                else:
                    response+=" (Same as last selling price)"

        else:
            response = "Selling price must be a number..."
    else:
        response = "Please include selling price after command name"

    return response

def buyCmd(message):
    print("Log buying price")
    response = ""
    splitmsg = message.content.split(" ")
    if len(splitmsg)>1:
        price = str(splitmsg[1])
        if price.isnumeric():
            savePrice(price,"buy",message.author.id)
            response = "Logged buying price of "+str(price)+" for user "+str(message.author)
        else:
            response = "Buying price must be a number..."
    else:
        response = "Please include buying price after command name"

    return response


def formatPrice(price):
    priceFmt = ""

    if price[0]=="buy":
        priceFmt="Can buy turnips from Daisy Mae for "
    else:
        priceFmt="Can sell turnips to Timmy & Tommy for "

    priceFmt+=str(price[1])+" bells at "
    
    priceTime = time.gmtime(float(price[2]))
    priceFmt+=time.strftime("%a, %d %b %Y %H:%M:%S UTC",priceTime)

    return priceFmt

def historyCmd(message):
    print("History command")
    response = ""

    splitmsg = message.content.split(" ")
    lookupName = ""

    if len(splitmsg) > 1:
       #History for specific user
       lookupName = splitmsg[1]
    else:
       #personal history
       lookupName = str(message.author)

    histUserName = lookupUserName(message,lookupName)
    histUserId = lookupIdByName(message,histUserName)
    priceList=loadPricesToList(str(histUserId)+".txt")
    response = "__**Historical info for "+histUserName+"**__"
    for price in priceList:
        response+="\n> "
        response+=formatPrice(price)

    return response

def oopsCmd(message):
    print("Oops command")
    response = ""

    splitmsg = message.content.split(" ")
    lookupName = ""

    if len(splitmsg) > 1:
       #History for specific user
       lookupName = splitmsg[1]
    else:
       #personal history
       lookupName = str(message.author)

    histUserName = lookupUserName(message,lookupName)
    histUserId = lookupIdByName(message,histUserName)
    fileName = str(histUserId)+".txt"
    priceList=loadPricesToList(str(histUserId)+".txt")
    response = "**Deleting last logged price for "+histUserName+"**"
    priceList=priceList[:-1]
    savePriceListToFile(fileName,priceList)

    return response

def clearCmd(message):
    print("Clearing history for "+str(message.author))
    prices = []
    userId = lookupIdByName(message,str(message.author))
    fileName = str(userId)+".txt" 
    savePriceListToFile(fileName,prices)
    return "**Cleared history for "+str(message.author)+"**"

def bestSellCmd(message):
    print("Find the best selling price in the last 12 hours")
    minTimeStamp = time.time() - (60.0*60*12) #Any timestamp larger than this is recent
    response = ""

    recentSellPrices = []

    for filename in os.listdir("prices"):
        f = open(priceFolder+"/"+filename,'r')
        mostRecentSell = ""
        for line in f:
            if line.startswith("sell"):
                mostRecentSell = line
        f.close()

        if mostRecentSell!="":
            sellInfo = mostRecentSell.strip().split(" ")
            if float(sellInfo[2]) >= minTimeStamp:
                userId = filename.strip(".txt")
                recentSellPrices.append([lookupNameById(message,userId),int(sellInfo[1]),sellInfo[2]])

    recentSellPrices.sort(key = lambda x: x[1],reverse=True)
    
    response = "__**Best Selling Prices in the Last 12 Hours**__"
    for price in recentSellPrices:
        sellTime = time.gmtime(float(price[2]))
        sellTimeStr=time.strftime("%a, %d %b %Y %H:%M:%S UTC",sellTime)
        response+="\n> "+str(price[0])+": "+str(price[1])+" bells at "+str(sellTimeStr)
    return response

def bestBuyCmd(message):
    print("Find the best buying price in the last 12 hours")
    minTimeStamp = time.time() - (60.0*60*12) #Any timestamp larger than this is recent
    response = ""

    recentBuyPrices = []

    for filename in os.listdir("prices"):
        f = open(priceFolder+"/"+filename,'r')
        mostRecentBuy = ""
        for line in f:
            if line.startswith("buy"):
                mostRecentBuy = line
        f.close()

        if mostRecentBuy!="":
            buyInfo = mostRecentBuy.strip().split(" ")
            if float(buyInfo[2]) >= minTimeStamp:
                userId = filename.strip(".txt")
                recentBuyPrices.append([lookupNameById(message,userId),int(buyInfo[1]),buyInfo[2]])

    recentBuyPrices.sort(key = lambda x: x[1])
    
    response = "__**Best Buying Prices in the Last 12 Hours**__"
    for price in recentBuyPrices:
        buyTime = time.gmtime(float(price[2]))
        buyTimeStr=time.strftime("%a, %d %b %Y %H:%M:%S UTC",buyTime)
        response+="\n> "+str(price[0])+": "+str(price[1])+" bells at "+str(buyTimeStr)
    return response

def generateGraph(message, user,graphname):
    plt.figure(figsize=(10,12),dpi=100)
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(twelvehour)
    ax.xaxis.set_major_formatter(hours_fmt)
    ax.xaxis.set_minor_locator(hours)

    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))
    legendElems = []

    if user!="": #Single lookup
        userId = lookupIdByName(message,user)
        userName = lookupNameById(message,userId)
        legendElems.append(userName)
        priceList = loadPricesToList(str(userId)+".txt")
        dates = []
        prices = []
        for price in priceList:
            if price[0]=="sell":
                prices.append(int(price[1]))
                dates.append(datetime.fromtimestamp(float(price[2])))

        if len(dates)>0:
            plt.plot(dates,prices,linestyle="-",marker='o',label=userName)

    else: #Lookup all
        for filename in os.listdir("prices"):
            userId = filename.strip(".txt")
            userName = lookupNameById(message,userId)
            legendElems.append(userName)
            priceList = loadPricesToList(str(userId)+".txt")
            dates = []
            prices = []
            for price in priceList:
                if price[0]=="sell":
                    prices.append(int(price[1]))
                    dates.append(datetime.fromtimestamp(float(price[2])))
            if len(dates)>0:
                plt.plot(dates,prices,linestyle="-",marker='o',label=userName)
    
    plt.xticks(rotation=45,ha="right",rotation_mode="anchor")
    plt.subplots_adjust(left=0.05,bottom=0.2,right=0.85)
    plt.grid(b=True,which='major',color='#666666',linestyle='-')
    ax.yaxis.grid(b=True,which='minor',color='#555555',linestyle=':')
    plt.ylabel("Price")
    plt.xlabel("Time (Eastern)")
    plt.title("Selling Prices")
    plt.legend(legendElems,loc="upper left",bbox_to_anchor=(1,1))

    figure = plt.gcf()
    figure.set_size_inches(18,9)

    plt.savefig(graphname,dpi=100)
    plt.close('all')
    
    return True


def graphCmd(message):
    print("Generate graph")
    response = "__**Historical Graph for "

    splitmsg = message.content.split(" ")
    lookupName = ""

    if len(splitmsg) > 1:
       #History for specific user
       lookupName = splitmsg[1]
       userId = lookupIdByName(message,lookupName)
       if userId == None:
           response = "No user found matching name "+lookupName
           if os.path.exists(graphcmdFile):
               #Remove any lingering file so that it doesn't get
               #attached to the message
               os.remove(graphcmdFile)
           return response

       #If we found the user ID, we SHOULD be able to find the name
       #since the user is likely still in the server.
       #Theoretically there is a very low probability race-condition here
       #if the user left the server between the userID lookup and the name
       #lookup, but I think that is so unlikely I don't want to bother
       #handling it.  I probably should have just put the check in here
       #instead of typing out this long comment.
       userName = lookupNameById(message,userId)
       response+=userName+"**__"
    else:
        response+="All Users**__"

    if generateGraph(message, lookupName,graphcmdFile):
        return response
    else:
        response+="\n Could not generate graph"
    
    return response

def turnipPatternLookup(xval):
    patterns = []
    print("Looking up turnip pattern.  X Val is "+str(xval))
    if xval>=0.91:
        patterns = [1,4]
    elif xval>=0.85:
        patterns = [2,3,4]
    elif xval >=0.80:
        patterns = [3,4]
    elif xval >=0.60:
        patterns = [1,4]
    else:
        patterns = [4]

    return patterns

def turnippatternCmd(message):
    print("Calculate your turnip pattern")
    response = ""

    splitmsg = message.content.split(" ")

    if len(splitmsg) < 3 or len(splitmsg) > 3:
        response = "Please provide Daisy Mae's price and your Monday morning price\n"
        response+= "eg. !turnippattern <buy price> <Monday morning sell price>"
    else:
        buyprice = splitmsg[1]
        mondayprice = splitmsg[2]

        if not buyprice.isnumeric() or not mondayprice.isnumeric():
            response = "Prices must be numbers"
        else:
            buyprice = int(buyprice)
            mondayprice = int(mondayprice)

            xval = float(mondayprice)/float(buyprice)
            patterns = turnipPatternLookup(xval)

            response = "Based on your prices, you will see one of the following patterns this week:\n"

            if 1 in patterns:
                response+="> **Random**: Prices are completely random.  Sell when it goes over your buying price\n"
            if 2 in patterns:
                response+="> **Decreasing**: Prices will continuously fall\n"
            if 3 in patterns:
                response+="> **Small Spike**: Prices fall until a spike occurs.  The price will go up three more times.  Sell on the third increase for maximum profit.  Spikes only occur from Monday to Thursday.\n"
            if 4 in patterns:
                response+="> **Big Spike**: Prices fall until a small spike.  Prices then decrease before shooting up twice.  Sell the second time prices shoot up after the decrease for maximum profit.  Spikes only occur from Monday to Thursday.\n"

    return response



def resetCmd(message):
    generateGraph(message,"",lastweekFile)
    for filename in os.listdir(priceFolder):
        mostrecentbuy = None
        prices = loadPricesToList(filename)
        for price in prices:
            if price[0]=="buy":
                mostrecentbuy = price

        if mostrecentbuy:
            savePriceListToFile(filename,[mostrecentbuy])
        else:
            os.remove(priceFolder+"/"+filename)
        

    return "Resetting data for a new week!"

def lastweekCmd(message):
    if os.path.exists(lastweekFile):
        return "__**Historical Graph from Last Week**__"
    else:
        return "No graph from last week"

def formatFossilList(fossils):
    response = ""
    for fossil in fossils:
        response+=fossil+", "

    if len(fossils)!=0:
        response = response[:-2]
    else:
        response = "No fossils"

    return response

def allfossilsCmd(message):
    print("List all possible fossils")
    response = "__**All Possible Fossils**__\n"
    response+= ">>> "
    response+=formatFossilList(baseFossilList)

    return response

def listfossilsCmd(message):
    print("List all remaining fossils for user")
    username = ""
    userid = 0

    splitmsg = message.content.split(" ")

    if len(splitmsg) > 1:
       #History for specific user
       lookupName = splitmsg[1]
       username = lookupUserName(message,lookupName)
       userid = lookupIdByName(message,username)
    else:
        username = str(message.author)
        userid = message.author.id

    fossils = loadFossilFileToList(str(userid)+".txt")
    fossilCount = len(fossils)
    response = "__**"+str(fossilCount)+" Fossils remaining for "+str(username)+"**__\n"
    response+=">>> "
    response+=formatFossilList(fossils)

    return response

def collectedfossilsCmd(message):
    print("List all the fossils user has collected")
    username = ""
    userid = 0

    splitmsg = message.content.split(" ")

    if len(splitmsg) > 1:
       #History for specific user
       lookupName = splitmsg[1]
       username = lookupUserName(message,lookupName)
       userid = lookupIdByName(message,username)
    else:
        username = str(message.author)
        userid = message.author.id

    remainingfossils = loadFossilFileToList(str(userid)+".txt")
    
    fossils = list(set(baseFossilList)-set(remainingfossils))
    
    fossilCount = len(fossils)
    response = "__**"+str(fossilCount)+" Fossils donated by "+str(username)+"**__\n"
    response+=">>> "
    response+=formatFossilList(fossils)

    return response

def collectCmd(message):
    print("Mark a fossil as collected by a user")
    response = ""
    splitmsg = message.content.split(" ")
    
    if len(splitmsg) == 1:
        response = "Please provide the name of a fossil to mark as collected"
    
    else:
        collectedfossilstr = " ".join(splitmsg[1:])
        collectedfossils = collectedfossilstr.split(",")
        strippedfossils = []
        for fossil in collectedfossils:
            strippedfossils.append(fossil.strip().lower())
        
        valid = []
        invalid = []
        dupes = []
        
        for fossil in strippedfossils:
            if fossil in baseFossilList:
                valid.append(fossil)
            else:
                invalid.append(fossil)
        
        if len(valid)>0:
            filename = str(message.author.id)+".txt"
            fossils = loadFossilFileToList(filename)
            for fossil in valid:
                if fossil in fossils:
                    fossils.remove(fossil)
                else:
                    dupes.append(fossil)
            saveFossilListToFile(filename,fossils)

        if len(dupes)>0:
            for dupe in dupes:
                valid.remove(dupe)

        if len(valid)>0:
            response+="Marked the following fossils as collected: \n"
            response+="> "
            response+=formatFossilList(valid)
            response+="\n"

        if len(dupes)>0:
            response+="The following fossils had already been collected: \n"
            response+="> "
            response+=formatFossilList(dupes)
            response+="\n"

        if len(invalid)>0:
            response+="Did not recognize the following fossils: \n"
            response+="> "
            response+=formatFossilList(invalid)
            response+="\n"

    return response

def uncollectCmd(message):
    print("Unmark a fossil as collected by a user")
    response = ""
    splitmsg = message.content.split(" ")
    
    if len(splitmsg) == 1:
        response = "Please provide the name of a fossil to unmark as collected"
    
    else:
        uncollectedfossilstr = " ".join(splitmsg[1:])
        uncollectedfossils = uncollectedfossilstr.split(",")
        strippedfossils = []
        for fossil in uncollectedfossils:
            strippedfossils.append(fossil.strip().lower())
        
        valid = []
        invalid = []
        dupes = []
        
        for fossil in strippedfossils:
            if fossil in baseFossilList:
                valid.append(fossil)
            else:
                invalid.append(fossil)
        
        if len(valid)>0:
            filename = str(message.author.id)+".txt"
            fossils = loadFossilFileToList(filename)
            for fossil in valid:
                if fossil not in fossils:
                    fossils.append(fossil)
                else:
                    dupes.append(fossil)
            saveFossilListToFile(filename,fossils)

        if len(dupes)>0:
            for dupe in dupes:
                valid.remove(dupe)

        if len(valid)>0:
            response+="Unmarked the following fossils as collected: \n"
            response+="> "
            response+=formatFossilList(valid)
            response+="\n"

        if len(dupes)>0:
            response+="The following fossils were already marked as not collected: \n"
            response+="> "
            response+=formatFossilList(dupes)
            response+="\n"

        if len(invalid)>0:
            response+="Did not recognize the following fossils: \n"
            response+="> "
            response+=formatFossilList(invalid)
            response+="\n"

    return response

def fossilsearchCmd(message):
    print("Search for users who haven't got specific fossils")
    response = ""
    splitmsg = message.content.split(" ")
    
    if len(splitmsg) == 1:
        response = "Please provide the name of a fossil to lookup users that don't have it"
    
    else:
        searchfossilstr = " ".join(splitmsg[1:])
        searchfossils = searchfossilstr.split(",")
        strippedfossils = []
        for fossil in searchfossils:
            strippedfossils.append(fossil.strip().lower())
        
        valid = []
        invalid = []
        
        #remove duplicates by converting to a set and back
        strippedfossils = list(set(strippedfossils))
        
        for fossil in strippedfossils:
            if fossil in baseFossilList:
                valid.append(fossil)
            else:
                invalid.append(fossil)
        
        if len(valid)>0:
            response+="__**Fossil Search**__\n"
            needed = False
            for filename in os.listdir(fossilFolder):
                fossils = loadFossilFileToList(filename)
                needs = []
                for fossil in valid:
                    if fossil in fossils:
                        needs.append(fossil)
                if len(needs)!=0:
                    needed = True
                    username = lookupNameById(message,filename.strip(".txt"))
                    response+="> "+username+" needs: "
                    response+= formatFossilList(needs)
                    response+="\n"

            if not needed:
                response+="No one currently needs this"

    return response

def fossilcountCmd(message):
    print("Provide fossil counts for users")
    response = ""

    splitmsg = message.content.split(" ")

    if len(splitmsg) == 1:
        response = "Please provide at least one user name to search for a fossil count"
    else:
        searchuserstr = " ".join(splitmsg[1:])
        searchusers = searchuserstr.split(",")
        
        valid = []
        invalid = []

        for user in searchusers:
            searchname = user.strip()
            username = lookupUserName(message,searchname)
            if username == None:
                invalid.append(searchname)
            else:
                userid = lookupIdByName(message,username)
                valid.append((username,userid))

        if len(valid)>0:
            response+="__**Fossil Count**__\n"
            for user in valid:
                fossils = loadFossilFileToList(str(user[1])+".txt")
                response+="> **"+user[0]+"** has "+str(len(fossils))+" fossils remaining\n"
            response+="\n"

        if len(invalid)>0:
            response+="__**Did not recognize the following names**__\n"
            for user in invalid:
                response+="> "+user+"\n"

    return response

def fossilhelpCmd(message):
    print("Fossil help")
    response = "__**Turbot Fossil Help!**__"
    response+= "\n> **!allfossils**"
    response+= "\n>    Shows all possible fossils that you can donate to the museum"
    response+= "\n> "
    response+= "\n> **!listfossils [user]**"
    response+= "\n>    Lists all fossils that you still need to donate.  If a user is provided, it gives the same information for that user instead"
    response+= "\n> "
    response+= "\n> **!collectedfossils [user]**"
    response+= "\n>    Lists all fossils that you have already donated.  If a user is provided, it gives the same information for that user instead"
    response+= "\n> "
    response+= "\n> **!collect <list of fossils>**"
    response+= "\n>    Mark fossils as donated to your museum.  The names must match the in-game item name, and more than one can be provided if separated by commas"
    response+= "\n> "
    response+= "\n> **!uncollect <list of fossils>**"
    response+= "\n>    Unmark fossils as donated to your museum.  The names must match the in-game item name, and more than one can be provided if separated by commas"
    response+= "\n> "
    response+= "\n> **!fossilsearch <list of fossils>**"
    response+= "\n>    Searches all users to see who needs the listed fossils.  The names must match the in-game item name, and more than one can be provided if separated by commas"
    response+= "\n> "
    response+= "\n> **!fossilcount <list of users>**"
    response+= "\n>    Provides a count of the number of fossils remaining for the comma-separated list of users"
    return response

def helpCmd(message):
    print("Help Command")
    response = "__**Turbot Help!**__"
    response+= "\n> **!help**"
    response+= "\n>    Shows this help screen"
    response+= "\n> "
    response+= "\n> **!fossilhelp**"
    response+= "\n>    Shows the help screen for fossil tracking"
    response+= "\n> "
    response+= "\n> **!sell <price>**"
    response+= "\n>    Log the price that you can sell turnips for on your island."
    response+= "\n> "
    response+= "\n> **!buy <price>**"
    response+= "\n>    Log the price that you can buy turnips from Daisy Mae on your island."
    response+= "\n> "
    response+= "\n> **!history [user]**"
    response+= "\n>    Show the historical turnip prices for a user.  If no user is specified, it will display your own prices."
    response+= "\n> "
    response+= "\n> **!clear**"
    response+= "\n>    Clears all of your own historical turnip prices."
    response+= "\n> "
    response+= "\n> **!oops [user]**"
    response+= "\n>    Remove the last logged turnip price.  If no user is specified, it will clear your own last logged price"
    response+= "\n> "
    response+= "\n> **!graph [user]**"
    response+= "\n>    Generates a graph of turnip prices for all users.  If a user is specified, only graph that users prices"
    response+= "\n> "
    response+= "\n> **!bestsell**"
    response+= "\n>    Finds the best (and most recent) selling prices logged in the last 12 hours."
    response+= "\n> "
    response+= "\n> **!bestbuy**"
    response+= "\n>    Finds the best (and most recent) buying prices logged in the last 12 hours."
    response+= "\n> "
    response+= "\n> **!lastweek**"
    response+= "\n>    Displays the final graph from the last week before the data was reset"
    response+= "\n> "
    response+= "\n> **!turnippattern <Sunday Buy Price> <Monday Morning Sell Price>**"
    response+= "\n>    Calculates the patterns you will see in your shop based on Daisy Mae's price on your island and your Monday morning sell price"
    response+= "\n> "
    response+= "\n> **!reset**"
    response+= "\n>    DO NOT USE UNLESS ASKED.  Generates a final graph for use with !lastweek and resets all data for all users"
    response+= "\n> "
    response+= "\n> turbot created by TheAstropath"

    return response


@client.event
async def on_ready():
    print("Logged in as {0.user}".format(client))

@client.event
async def on_message(message):
    response = None
    attachment = None
    
    if str(message.channel.type) == "text":
        if message.channel.name not in authChannels:
            return
    else:
        print("Message received on channel type: "+str(message.channel.type))
        return
    
    if message.author == client.user:
        return

    if message.content.startswith("!hello"):
        response = helloCmd(message)

    if message.content.startswith("!lookup"):
        response = lookupCmd(message)
    
    if message.content.startswith("!sell"):
        response = sellCmd(message)

    if message.content.startswith("!buy"):
        response = buyCmd(message) 

    if message.content.startswith("!help"):
        response = helpCmd(message)

    if message.content.startswith("!fossilhelp"):
        response = fossilhelpCmd(message) 

    if message.content.startswith("!history"):
        response = historyCmd(message)

    if message.content.startswith("!bestsell"):
        response = bestSellCmd(message)

    if message.content.startswith("!bestbuy"):
        response = bestBuyCmd(message) 

    if message.content.startswith("!graph"):
        #Graph generation isn't instantaneous,
        #so show that the bot is typing for
        #some feedback
        async with message.channel.typing():
            response = graphCmd(message)
            if os.path.exists(graphcmdFile):
                attachment = discord.File(graphcmdFile)

    if message.content.startswith("!clear"):
        response = clearCmd(message)

    if message.content.startswith("!oops"):
        response = oopsCmd(message)
    
    if message.content.startswith("!turnippattern"):
        response = turnippatternCmd(message)

    if message.content.startswith("!reset"):
        response = resetCmd(message)
    
    if message.content.startswith("!lastweek"):
        response = lastweekCmd(message)
        if os.path.exists(lastweekFile):
            attachment = discord.File(lastweekFile)

    if message.content.startswith("!allfossils"):
        response = allfossilsCmd(message)

    if message.content.startswith("!listfossils"):
        response = listfossilsCmd(message)

    if message.content.startswith("!collect"):
        response = collectCmd(message)

    if message.content.startswith("!uncollect"):
        response = uncollectCmd(message)

    if message.content.startswith("!collectedfossils"):
        response = collectedfossilsCmd(message)

    if message.content.startswith("!fossilsearch"):
        response = fossilsearchCmd(message)

    if message.content.startswith("!fossilcount"):
        response = fossilcountCmd(message)


    if response != None:
        await message.channel.send(response,file=attachment)

client.run(token)

