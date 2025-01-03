from geosolver import geoserver_interface
questions = geoserver_interface.download_questions('test')
for id_, question in questions.iteritems():
  print(question.text)
