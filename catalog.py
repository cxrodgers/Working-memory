# Unit Catalog

# Decided to use an SQLite database for this

import os
import sqlite3

def UnitCatalog(object):
    
    def __init__(self, db_filename):

        db_is_new = not os.path.exists(db_filename)

        conn = sqlite3.connect(db_filename)

        if db_is_new:
            print 'Need to create schema'
        else:
            print 'Database exists, assume schema does, too.'

        conn.close()