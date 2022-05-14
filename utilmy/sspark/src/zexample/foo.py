from pyspark.sql.functions import udf
import httplib, urllib, base64, json 
def whichLanguage(text):
    headers = {
        # Request headers
        'Content-Type': 'application/json',
        'Ocp-Apim-Subscription-Key': '{your subscription key here}',
    }

    params = urllib.urlencode({
        # Request parameters
        'numberOfLanguagesToDetect': '1',
    })

  
    # cons up the payload
    body = {'documents' : [{'id':'23','text':text}]}
    bodyj = json.dumps(body)

    try:
        conn = httplib.HTTPSConnection('westus.api.cognitive.microsoft.com')
        conn.request("POST", "/text/analytics/v2.0/languages?%s" % params, bodyj, headers)
        response = conn.getresponse()
        data = response.read()
        s = data.decode()
        x = json.loads(s) 
        conn.close()
        # todo: could return more interesting stuff here 
        return x['documents'][0]['detectedLanguages'][0]['name']
    except Exception as e:
        print("[Errno {0}] {1}".format(e.errno, e.strerror))
        return 'foo'

udfWhichLanguage = udf(whichLanguage, StringType())



df.withColumn('newStuff',udfWhichLanguage('{your column name here}')).show(50)