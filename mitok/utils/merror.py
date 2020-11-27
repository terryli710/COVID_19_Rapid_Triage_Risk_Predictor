class MError(object):
    
    E_FIELD_BASE = 0
    E_FIELD_IMAGE = 1
    E_FIELD_DICOM = 2
    E_FIELD_SERIES = 3

    E_FIELD_NAME = {
            E_FIELD_BASE:'Base', 
            E_FIELD_IMAGE:'Image', 
            E_FIELD_DICOM:'Dicom', 
            E_FIELD_SERIES: 'Series',
                   }

    def __init__(self, e_field=0, e_no= "", e_msg=""):
        self.e_field = e_field
        self.e_no = e_no
        self.e_msg = e_msg

    def check(self):
        if self.e_no == 0 or self.e_no == "":
            return True
        else:
            return False

    def message(self):
        return self.e_msg

    def pretty_print(self):
        if self.e_no == 0:
            print('[Info:%s] %s' % (MError.E_FIELD_NAME[self.e_field], self.e_msg))
        else:
            print('[Error:%s:%s] %s' % (MError.E_FIELD_NAME[self.e_field], self.e_no, self.e_msg))

