def extractNounPhrase(text, precedingPOS = ["NOUN"], sep = "_"): # Extract noun phrases
  doc = nlp(text)
  patterns = []
  for POS in precedingPOS: # POS stands for part of speech
    patterns.append(f"POS:{POS} POS:NOUN:+") # look for phrases starting with selected POS and ending with one or more nouns
  matches = textacy.extract.matches.token_matches(doc, patterns = patterns)
  nounPhrase = [sep.join([j.lemma_ for j in i]) for i in matches]
  return nounPhrase
udfNounPhrase = udf(extractNounPhrase, ArrayType(StringType())) # Wrap function in udf to be applied on Spark dataframe