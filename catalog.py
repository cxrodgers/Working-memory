# Unit Catalog

# Decided to use SQLAlchemy for this

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, Float, String
from sqlalchemy import ForeignKey, create_engine
from sqlalchemy.orm import relationship, backref, sessionmaker

#################################################
# Make all your classes here

Base = declarative_base()

class Project(Base):
    __tablename__ = 'projects'
    
    id = Column(Integer, primary_key=True)
    project = Column(String)
    researcher = Column(String)
    
    units = relationship("Unit", order_by="Unit.id", backref="project")
    
    def __init__(self, project, researcher = None):
        self.project = project
        self.researcher = researcher
    
    def __repr__(self):
        return "<Project('%s')>" % self.project

class Unit(Base):
    '''
    project : which project/task the unit was recorded for
    rat : the name of the rat the unit came from
    date : date unit was recorded
    tetrode : tetrode number unit was recorded on
    cluster : cluster number of the unit from the sorting
    datapath : path to the folder containing all the data
    falsePositive : false positive rate, from cluster metrics
    falseNegative : false negative rate, from cluster metrics
    notes : any notes about the unit
    depth : depth at which the unit was recorded
    
    '''
    __tablename__ = 'units'
    
    id = Column(Integer, primary_key=True)
    
    rat = Column(String)
    date = Column(String)
    tetrode = Column(Integer)
    cluster = Column(Integer)
    datapath = Column(String)
    falsePositive = Column(Float)
    falseNegative = Column(Float)
    notes = Column(String)
    depth = Column(Float)
    
    project_id = Column(Integer, ForeignKey("projects.id"))
    #project = relationship("Project", backref=backref('units', order_by=id))
    
    def __init__(self, ratname, date, tetrode, cluster):
        self.rat = ratname
        self.date = date
        self.tetrode = tetrode
        self.cluster = cluster
    
    def __repr__(self):
        return "<Unit('%s', '%s', tetrode '%s', cluster '%s')>" \
            % (self.ratname, self.date, self.tetrode, self.cluster)


#################################################



def get_session(database, echo = False):
    ''' Connects to the database and returns the SQLAlchemy session'''
    
    engine = create_engine('sqlite://'+database, echo = echo)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # If the tables don't exist in the database, this will create them
    Base.metadata.create_all(engine) 
    
    return session
    
def start_project(session, project, researcher = None):
    ''' Start a project and add it to the database '''
    
    session.add(Project(project, researcher))
    session.commit()

    
def add_unit(project, ratname, date, tetrode, cluster, **kwargs):
    ''' Add a unit to a project '''
    
    # Query for project and rat
    
    # If the project doesn't exit, raise an exception
    
    # If rat doesn't exist, add rat to project
    # Make it case insensitive, all uppercase
    
    # Add unit to rat

    
    pass

def update_unit(unit_id, **kwargs):
    ''' Update a unit already in the database '''
    
    pass

