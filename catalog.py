# Unit Catalog

# Decided to use SQLAlchemy for this

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.exc import *
from sqlalchemy import Column, Integer, Float, String
from sqlalchemy import ForeignKey, create_engine, and_
from sqlalchemy.orm import relationship, backref, sessionmaker

#################################################
# Make all your classes here

Base = declarative_base()

class Project(Base):
    __tablename__ = 'projects'
    
    id = Column(Integer, primary_key=True)
    project = Column(String)
    researcher = Column(String)
    
    sessions = relationship("Session", order_by="Session.id", backref="project")
    
    def __init__(self, project, researcher = None):
        self.project = project
        self.researcher = researcher
    
    def __repr__(self):
        return "<Project('%s')>" % self.project

class Session(Base):
    '''
    project : which project/task the unit was recorded for
    rat : the name of the rat the unit came from
    date : date unit was recorded
    '''
    __tablename__ = 'sessions'
    
    id = Column(Integer, primary_key=True)
    
    rat = Column(String)
    date = Column(String)
    notes = Column(String)
    duration = Column(Float)
    
    project_id = Column(Integer, ForeignKey("projects.id"))
    
    units = relationship("Unit", order_by="Unit.id", backref="session")
    
    def __init__(self, project, rat, date):
        self.project = project
        self.rat = rat
        self.date = date
    
    def __repr__(self):
        return "<Session(%s, %s)>" % self.rat, self.date

class Unit(Base):
    '''
    
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

    tetrode = Column(Integer)
    cluster = Column(Integer)
    path = Column(String)
    falsePositive = Column(Float)
    falseNegative = Column(Float)
    notes = Column(String)
    depth = Column(Float)
    
    session_id = Column(Integer, ForeignKey("sessions.id"))
    
    def __init__(self, session, tetrode, cluster):
        self.session = session
        self.tetrode = tetrode
        self.cluster = cluster
    
    def __repr__(self):
        return "<Unit(%s, %s,tetrode %s,cluster %s)>" \
            % (self.id, self.session, self.tetrode, self.cluster)


class Catalog(object):
    
    def __init__(self, database, echo = False):
        
        self.engine = create_engine('sqlite://'+database, echo = echo)
        self._ConnectGen = sessionmaker(bind=self.engine)
        
        Base.metadata.create_all(self.engine)
    
    def open(self):
        ''' Returns a connection to the database, an SQLAlchemy session generated
            by a sessionmaker bound to self.engine '''
        
        connection = self._ConnectGen()
        return connection
    
    def start_project(self, project, researcher = None):
        ''' Start a project and add it to the database '''
        
        conn = self.open()
        conn.add(Project(project, researcher))
        conn.commit()
        conn.close()
    
    def get_project(self, project, connection=None):
        ''' Returns a Project
        
        Parameters
        ----------
        project : string
            name of the project you want
        connection : session SQLAlchemy object
            pass in a session already opened to get the project from there, or
            leave it empty so that it will open a new session and return the project
            You can get this from self.open()
        
        Returns
        -------
        out : Project object
            the requested Project
        
        '''      
        
        close = False
        if connection == None:
            close = True
            conn = self.open()
        
        try:
            proj = conn.query(Project).filter(Project.project == project).one()
        except NoResultFound:
            raise Exception("Project %s doesn't exist, create it first." % project)
            
            
        if close:
            conn.close()
            
        return proj
    
    def get_session(self, rat, date, connection=None):
        ''' Returns a Session from the specified rat and date'''
        
        close = False
        if connection == None:
            close = True
            conn = self.open()
        
        
        try:
            session = conn.query(Session).filter(Session.rat==rat).\
                filter(Session.date==date).one()
        except NoResultFound:
            session = None
            
        
        if close:
            conn.close()
        
        return session
        
    
    def add_unit(self, project, rat, date, tetrode, cluster, **kwargs):
        ''' Add a unit to a project '''
        
        # First, we'll check if the unit already exists
        unit = self.get_unit(rat, date, tetrode, cluster)
        
        conn = self.open()
        
        proj = self.get_project(project)
        
        session = self.get_session(rat, date)
        if session == None:
            session = Session(proj, rate, date)
        
        unit = Unit(session, tetrode, cluster)
        conn.commit()
    
        #self.update_unit(unit.id, **kwargs)
        
        conn.close()
    
    def get_unit(self, rat, date, tetrode, cluster):
        '''Return the unit id specified by the project, rate, date, tetrode, cluster
            Should I just have it return the unit object???
        '''
        conn = self.open()
        session = self.get_session(rat, date)
        
        try:
            unit = conn.query(Unit).\
                filter(and_(Unit.session==session, Unit.tetrode==tetrode, 
                Unit.cluster==cluster))
        except NoResultFound:
            print "The unit doesn't exist"
        
        conn.close()
        return unit
        
    

    def update_unit(self, unit_id, **kwargs):
        ''' Update a unit already in the database '''
        
        # Doing it this way so that when you change the attributes of the
        # Unit class, the valid keys here will change automatically
        valid_keys = [key for key in dir(Unit) if key[0]!='_']
        excise = ['metadata', 'id', 'session', 'session_id']  # don't want these to be accessed
        for exc in excise:
            valid_keys.remove(exc)
            
        if kwargs:
            for key in kwargs.iterkeys():
                if key in valid_keys:
                    pass
                else:
                    raise NameError('%s not a valid keyword' % key)
        
        
        pass

