import io

import h5py
import numpy as np
import PIL.Image


def hdf5_join(a, b):
    a = a.rstrip('/')
    b = b.lstrip('/')
    return a + '/' + b


# Image
#   version     ATTRIB      Integer     1
#   width       ATTRIB      Integer
#   height      ATTRIB      Integer
#   seed        ATTRIB      Integer
#   format      ATTRIB      String
#   data        ATTRIB      np.void(binary_blob)        .tostring() to restore

class Image:
    def __init__(self, width=0, height=0, seed=-1, format='', data=b'', version=1):
        self.version = version
        self.width = width
        self.height = height
        self.seed = seed
        self.format = format
        self.data = data


    @classmethod
    def fromhdf5(cls, file, root=u'/'):
        group = file[root]
        assert group.attrs['version'] == 1
        return cls(version=1, width=group.attrs['width'],
            height=group.attrs['height'], seed=group.attrs['seed'],
            format=group.attrs['format'], data=group['data'][0])


    @classmethod
    def fromPIL(cls, pil_image, seed):
        buf = io.BytesIO()
        pil_image.save(buf, format='JPEG')
        return cls(version=1, width=pil_image.size[0],
            height=pil_image.size[1], seed=seed,
            format='JPEG', data=buf.getvalue())


    def tohdf5(self, file, root=u'/'):
        file.require_group(root)
        group = file[root]
        group.attrs['version'] = 1
        group.attrs['width'] = self.width
        group.attrs['height'] = self.height
        group.attrs['seed'] = self.seed
        group.attrs['format'] = self.format
        datatype = h5py.special_dtype(vlen=np.dtype('uint8'))
        if 'data' in group:
            del group['data']
        data_group = group.create_dataset('data', (1,), dtype=datatype)
        data_group[0] = np.frombuffer(self.data, dtype='uint8')


    def toPIL(self):
        buf = io.BytesIO(self.data)
        return PIL.Image.open(buf)



# Face
#   version     ATTRIB      Integer     1
#   name        ATTRIB      String
#   desc        ATTRIB      String
#   latents     Dataset     float       512-d
#   thumbnail   Group       Image

class Face:
    def __init__(self, name='', desc='', latents=[0.0*512], psi=0.0, thumbnail=None, version=1):
        self.version = version
        self.name = name
        self.desc = desc
        self.latents = latents
        self.psi = psi
        self.thumbnail = thumbnail if thumbnail != None else Image()


    @classmethod
    def fromhdf5(cls, file, root=u'/'):
        group = file[root]
        assert group.attrs['version'] == 1
        return cls(version=1, name=group.attrs['name'], desc=group.attrs['desc'],
            latents=group['latents'][:].tolist(), psi=group.attrs['psi'],
            thumbnail=Image.fromhdf5(file, hdf5_join(root, u'thumbnail')))


    def tohdf5(self, file, root=u'/'):
        file.require_group(root)
        group = file[root]
        group.attrs['version'] = 1
        group.attrs['name'] = self.name
        group.attrs['desc'] = self.desc
        group.attrs['psi'] = self.psi
        group.require_dataset('latents', shape=(512,), dtype='f16', exact=True)
        if 'latents' in group:
            del group['latents']
        group['latents'] = np.array(self.latents)
        self.thumbnail.tohdf5(file, hdf5_join(root, u'thumbnail'))



# PyroParams
#   version     ATTRIB      Integer     1
#   pyroversion ATTRIB      String
#   paramstore  ATTRIB      np.void(binary_blob)

class PyroParams:
    def __init__(self, pyroversion='', paramstore=b'', version=1):
        self.version = version
        self.pyroversion = pyroversion
        self.paramstore = paramstore


    @classmethod
    def fromhdf5(cls, file, root=u'/'):
        group = file[root]
        assert group.attrs['version'] == 1
        return cls(version=1, pyroversion=group.attrs['pyroversion'],
            paramstore=group['paramstore'][0].tobytes())


    def tohdf5(self, file, root=u'/'):
        file.require_group(root)
        group = file[root]
        group.attrs['version'] = 1
        group.attrs['pyroversion'] = self.pyroversion
        datatype = h5py.special_dtype(vlen=np.dtype('uint8'))
        if 'paramstore' in group:
            del group['paramstore']
        paramstore_group = group.create_dataset('paramstore', (1,), dtype=datatype, compression='gzip')
        paramstore_group[0] = np.frombuffer(self.paramstore, dtype='uint8')


# Model
#   version     ATTRIB      Integer     1
#   name        ATTRIB      String
#   desc        ATTRIB      String
#   type        ATTRIB      String      CdV
#   gentype     ATTRIB      String      StyleGAN
#   params      Group       PyroParams

class Model:
    def __init__(self, name='', desc='', modeltype='', gentype='', pyroparams=None, version=1):
        self.version = version
        self.name = name
        self.desc = desc
        self.modeltype = modeltype
        self.gentype = gentype
        self.pyroparams = pyroparams if pyroparams != None else PyroParams()


    @classmethod
    def fromhdf5(cls, file, root=u'/'):
        group = file[root]
        assert group.attrs['version'] == 1
        return cls(version=1, name=group.attrs['name'], desc=group.attrs['desc'],
            modeltype=group.attrs['modeltype'], gentype=group.attrs['gentype'],
            pyroparams=PyroParams.fromhdf5(file, hdf5_join(root, u'pyroparams')))


    def tohdf5(self, file, root=u'/'):
        file.require_group(root)
        group = file[root]
        group.attrs['version'] = 1
        group.attrs['name'] = self.name
        group.attrs['desc'] = self.desc
        group.attrs['modeltype'] = self.modeltype
        group.attrs['gentype'] = self.gentype
        self.pyroparams.tohdf5(file, hdf5_join(root, u'pyroparams'))



# version   ATTRIB      Integer     1
# n         ATTRIB      Integer
# 00000000  Group       Face
# 00000001  Group       Face
# ...

class FaceList:
    def __init__(self, faces=[], version=1):
        self.version = version
        self.faces = faces


    @classmethod
    def fromhdf5(cls, file, root=u'/'):
        group = file[root]
        assert group.attrs['version'] == 1
        n = group.attrs['n']
        faces = [Face.fromhdf5(file, hdf5_join(root, u'{:08}'.format(i))) for i in range(n)]
        return cls(version=1, faces=faces)


    def tohdf5(self, file, root=u'/'):
        file.require_group(root)
        group = file[root]
        group.attrs['version'] = 1
        group.attrs['n'] = len(self.faces)
        for existing_face in group.keys():
            if int(existing_face) > len(self.faces)-1:
                del group[existing_face]
        for i, face in enumerate(self.faces):
            face.tohdf5(file, hdf5_join(root, u'{:08}'.format(i)))



# SearchIterationData_SVI_ACCUM_ADAM_InOut
#   version     ATTRIB      Integer     1
#   lr          ATTRIB      Float
#   beta1       ATTRIB      Float
#   beta2       ATTRIB      Float
#   niter       ATTRIB      Integer
#   faces       Group       FaceList
#   in_indices  Dataset     Integer
#   out_indices Dataset     Integer

class SearchIterationData_SVI_ACCUM_ADAM_InOut:
    def __init__(self, lr=0.0, beta1=0.0, beta2=0.0, niter=0, faces=None, in_indices=[], out_indices=[], seed=-1, version=1):
        self.version = version
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.niter = niter
        self.faces = faces if faces != None else FaceList()
        self.in_indices = in_indices
        self.out_indices = out_indices
        self.seed = seed


    @classmethod
    def fromhdf5(cls, file, root=u'/'):
        group = file[root]
        assert group.attrs['version'] == 1
        faces = FaceList.fromhdf5(file, hdf5_join(root, u'faces'))
        in_indices = group['in_indices'][0].tolist()
        out_indices = group['out_indices'][0].tolist()
        return cls(version=1, lr=group.attrs['lr'], beta1=group.attrs['beta1'],
            beta2=group.attrs['beta2'], niter=group.attrs['niter'], faces=faces,
            in_indices=in_indices, out_indices=out_indices, seed=group.attrs['seed'])


    def tohdf5(self, file, root=u'/'):
        file.require_group(root)
        group = file[root]
        group.attrs['version'] = 1
        group.attrs['lr'] = self.lr
        group.attrs['beta1'] = self.beta1
        group.attrs['beta2'] = self.beta2
        group.attrs['niter'] = self.niter
        group.attrs['seed'] = self.seed
        index_type = h5py.special_dtype(vlen=np.dtype('uint32'))
        in_indices = group.create_dataset('in_indices', (1,), dtype=index_type, compression='gzip')
        in_indices[0] = self.in_indices
        out_indices = group.create_dataset('out_indices', (1,), dtype=index_type, compression='gzip')
        out_indices[0] = self.out_indices
        self.faces.tohdf5(file, hdf5_join(root, u'faces'))



# SearchIteration
#   version     ATTRIB      Integer     1
#   method      ATTRIB      String
#   input       Group       SearchIterationData_<method>
#   in_params   Group       PyroParams  Parameters *before* the search
#   out_params   Group      PyroParams  Parameters *after* the search

class SearchIteration:
    def __init__(self, method='', input=None, in_params=None, out_params=None, version=1):
        self.version = version
        self.method = method
        if method == 'SVI_ACCUM_ADAM_InOut':
            self.input = input if input != None else SearchIterationData_SVI_ACCUM_ADAM_InOut()
        else:
            assert False, 'Unknown search method {}'.format(method)
        self.in_params = in_params if in_params != None else PyroParams()
        self.out_params = out_params if out_params != None else PyroParams()


    @classmethod
    def fromhdf5(cls, file, root=u'/'):
        group = file[root]
        assert group.attrs['version'] == 1
        method = group.attrs['method']
        if method == 'SVI_ACCUM_ADAM_InOut':
            input = SearchIterationData_SVI_ACCUM_ADAM_InOut.fromhdf5(file, hdf5_join(root, u'input'))
        else:
            assert False, 'Unknown search method {}'.format(method)
        in_params = PyroParams.fromhdf5(file, hdf5_join(root, u'in_params'))
        out_params = PyroParams.fromhdf5(file, hdf5_join(root, u'out_params'))
        return cls(version=1, method=method, input=input, in_params=in_params, out_params=out_params)


    def tohdf5(self, file, root=u'/'):
        file.require_group(root)
        group = file[root]
        group.attrs['version'] = 1
        group.attrs['method'] = self.method
        self.input.tohdf5(file, hdf5_join(root, u'input'))
        self.in_params.tohdf5(file, hdf5_join(root, u'in_params'))
        self.out_params.tohdf5(file, hdf5_join(root, u'out_params'))



# SearchIterationList
#   version   ATTRIB      Integer     1
#   n         ATTRIB      Integer
#   00000000  Group       SearchIteration
#   00000001  Group       SearchIteration
#   ...

class SearchIterationList:
    def __init__(self, iterations=[], version=1):
        self.version = version
        self.iterations = iterations


    @classmethod
    def fromhdf5(cls, file, root=u'/'):
        group = file[root]
        assert group.attrs['version'] == 1
        n = group.attrs['n']
        iterations = [SearchIteration.fromhdf5(file, hdf5_join(root, u'{:08}'.format(i))) for i in range(n)]
        return cls(version=1, iterations=iterations)


    def tohdf5(self, file, root=u'/'):
        file.require_group(root)
        group = file[root]
        group.attrs['version'] = 1
        group.attrs['n'] = len(self.iterations)
        for existing_iter in group.keys():
            if int(existing_iter) > len(self.iterations)-1:
                del group[existing_iter]
        for i, iter in enumerate(self.iterations):
            iter.tohdf5(file, hdf5_join(root, u'{:08}'.format(i)))



# Search
#   version     ATTRIB      Integer     1
#   name        ATTRIB      String
#   desc        ATTRIB      String
#   modeltype   ATTRIB      String      CdV
#   gentype     ATTRIB      String      StyleGAN
#   iterations  Group       SearchIterationList

class Search:
    def __init__(self, name='', desc='', modeltype='', gentype='', iterations=None, version=1):
        self.version = version
        self.name = name
        self.desc = desc
        self.modeltype = modeltype
        self.gentype = gentype
        self.iterations = iterations if iterations != None else SearchIterationList()


    @classmethod
    def fromhdf5(cls, file, root=u'/'):
        group = file[root]
        assert group.attrs['version'] == 1
        return cls(version=1, name=group.attrs['name'], desc=group.attrs['desc'],
            modeltype=group.attrs['modeltype'], gentype=group.attrs['gentype'],
            iterations=SearchIterationList.fromhdf5(file, hdf5_join(root, u'iterations')))


    def tohdf5(self, file, root=u'/'):
        file.require_group(root)
        group = file[root]
        group.attrs['version'] = 1
        group.attrs['name'] = self.name
        group.attrs['desc'] = self.desc
        group.attrs['modeltype'] = self.modeltype
        group.attrs['gentype'] = self.gentype
        self.iterations.tohdf5(file, hdf5_join(root, u'iterations'))
