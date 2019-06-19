import logging
import sys
import uuid

import h5py
import PIL.Image
from PyQt5 import QtCore, QtWidgets, uic
from PyQt5.QtGui import QPixmap, QIcon, QImage
import torch

import cdv
import data
import generator


Ui_MainWindow, QtBaseClass = uic.loadUiType('ui/search.ui')
Ui_SaveModelDlg, QtBaseClass = uic.loadUiType('ui/save_model.ui')


class SaveModelDlg(QtWidgets.QDialog, Ui_SaveModelDlg):
    def __init__(self):
        QtWidgets.QDialog.__init__(self)
        Ui_SaveModelDlg.__init__(self)
        self.setupUi(self)
        self.actionBrowse.clicked.connect(self.onBrowse)


    def onBrowse(self):
        path, ret = QtWidgets.QFileDialog.getSaveFileName(self,
            'Save Model file', '', 'Model files (*.model)')
        if path != '':
            self.path.setText(path)


    def getValue(self):
        return self.path.text(), self.name.text(), self.description.toPlainText()



class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        self.protect_data = False  # Semaphore to prevent the
            # send_params_to_data signals having an effect if triggered by the
            # actions of load_params_from_data.
        self.onFileNew()
        self.update_ui_enabled_elements()

        self.set_good.clicked.connect(self.onSetGood)
        self.set_bad.clicked.connect(self.onSetBad)
        self.search_go.clicked.connect(self.onSearchGo)
        self.sample_go.clicked.connect(self.onSampleGo)

        self.good_images.itemClicked.connect(self.onGoodClicked)
        self.bad_images.itemClicked.connect(self.onBadClicked)

        # Connectors to update the data structure in response to user UI changes
        self.global_description.textChanged.connect(self.send_params_to_data)
        self.global_name.textChanged.connect(self.send_params_to_data)
        self.generator_psi.valueChanged.connect(self.send_params_to_data)
        self.generator_seed.valueChanged.connect(self.send_params_to_data)
        self.sample_count.valueChanged.connect(self.send_params_to_data)
        self.sample_seed.valueChanged.connect(self.send_params_to_data)
        self.search_beta1.valueChanged.connect(self.send_params_to_data)
        self.search_beta2.valueChanged.connect(self.send_params_to_data)
        self.search_iterations.valueChanged.connect(self.send_params_to_data)
        self.search_learningrate.valueChanged.connect(self.send_params_to_data)

        # Connectors to update the enabled status of UI elements in response
        # to user action
        self.good_images.model().rowsInserted.connect(self.update_ui_enabled_elements)
        self.good_images.model().rowsRemoved.connect(self.update_ui_enabled_elements)
        self.good_images.itemSelectionChanged.connect(self.update_ui_enabled_elements)
        self.bad_images.model().rowsInserted.connect(self.update_ui_enabled_elements)
        self.bad_images.model().rowsRemoved.connect(self.update_ui_enabled_elements)
        self.bad_images.itemSelectionChanged.connect(self.update_ui_enabled_elements)

        self.actionNew.triggered.connect(self.onFileNew)
        self.actionNewFromModel.triggered.connect(self.onFileNewFromModel)
        self.actionOpen.triggered.connect(self.onFileOpen)
        self.actionSave.triggered.connect(self.onFileSave)
        self.actionSaveAs.triggered.connect(self.onFileSaveAs)
        self.actionQuit.triggered.connect(self.onFileQuit)
        self.actionExportFace.triggered.connect(self.onFaceExport)
        self.actionExportModel.triggered.connect(self.onModelExport)

        self.list_rounds.currentItemChanged.connect(self.onRoundChange)

        QtWidgets.QShortcut(QtCore.Qt.Key_Up, self, self.onSetGood)
        QtWidgets.QShortcut(QtCore.Qt.Key_Down, self, self.onSetBad)

        self.good_images.setIconSize(QtCore.QSize(128, 128))
        self.bad_images.setIconSize(QtCore.QSize(128, 128))


    @staticmethod
    def search_to_comparisons(search):
        # Construct a dictionary of latents, and a list of comparison tuples,
        # that encode all InOut data for all search iterations.
        latents = {}
        comparisons = []
        latent_counter = 0      # A globally unique latent ID
        for i, iter in enumerate(search.iterations.iterations):
            assert iter.method == 'SVI_ACCUM_ADAM_InOut'
            this_round_latent_keys = []     # Mapping from round latent indices to the globally unique ID
            for face in iter.input.faces.faces:
                latents[latent_counter] = face.latents
                this_round_latent_keys.append(latent_counter)
                latent_counter += 1
            for in_index in iter.input.in_indices:
                for out_index in iter.input.out_indices:
                    comparisons.append((this_round_latent_keys[in_index], this_round_latent_keys[out_index], 1.))
        return (latents, comparisons)


    # TODO: Warning about breaking existing search
    def onSearchGo(self):
        if self.good_images.count() == 0 or self.bad_images.count() == 0:
            return

        niter = self.search_iterations.value()
        lr = self.search_learningrate.value()
        beta1 = self.search_beta1.value()
        beta2 = self.search_beta2.value()
        seed = 27182818
        cpuonly = not self.search_cuda.isChecked()

        # Get all latents and comparisons for all iterations of the search
        latents, comparisons = self.search_to_comparisons(self.search)

        # Get the starting 'input' parameter store.  Fallback to None (ie prior)
        # if necessary
        round = self.get_current_round()
        if round == None or round.in_params.paramstore == b'':
            paramstore = None
        else:
            paramstore = round.in_params.paramstore

        # Do the learning
        newparams = cdv.learn(paramstore, latents, comparisons, svi_iters=niter,
            lr=lr, beta1=beta1, beta2=beta2, seed=seed, cpuonly=cpuonly)

        # Add these learned parameters to the latest SearchIteration.  Update other
        # metadata fields as well.
        self.search.iterations.iterations[-1].out_params = data.PyroParams(paramstore=newparams)
        self.search.iterations.iterations[-1].input.lr = lr
        self.search.iterations.iterations[-1].input.beta1 = beta1
        self.search.iterations.iterations[-1].input.beta2 = beta2
        self.search.iterations.iterations[-1].input.niter = niter
        self.search.iterations.iterations[-1].input.seed = seed

        # Add a new SearchIteration stub.
        newinput = data.SearchIterationData_SVI_ACCUM_ADAM_InOut(lr=lr,
            beta1=beta1, beta2=beta2, niter=niter, seed=seed)
        newiter = data.SearchIteration(method='SVI_ACCUM_ADAM_InOut',
            input=newinput, in_params=data.PyroParams(paramstore=newparams))
        self.search.iterations.iterations.append(newiter)

        self.recreate_round_list()
        self.list_rounds.setCurrentRow(self.list_rounds.count()-1)
        self.set_modified(True)


    def onSampleGo(self):
        round = self.get_current_round()
        if round == None:
            return

        sample_seed = self.sample_seed.value()
        gen_psi = self.generator_psi.value()
        gen_seed = self.generator_seed.value()
        n = self.sample_count.value()
        cpuonly = not self.generator_cuda.isChecked()

        paramstore_in = round.in_params.paramstore
        model_name = ''
        if paramstore_in == b'':
            logging.info('Empty model encountered; falling back to prior')
            model_name = 'PRIOR'
            paramstore_in = None

        # TODO: There is currently an issue where the latents from CPU and CUDA
        # are completely different. Get around this for now by forcing CPU
        # sampling. This needs to be investigated though as there are also
        # subtle differences in the generator stage.
        latent_vectors = cdv.sample(paramstore_in, n, sample_seed, cpuonly=True)

        faces = [data.Face(name='sample_{}_{}'.format(model_name, str(uuid.uuid4())[:8]), latents=x, psi=gen_psi) for x in latent_vectors]
        for face in faces:
            face.thumbnail = data.Image.fromPIL(
                generator.latent_to_image([face.latents], seed=gen_seed,
                    psi=gen_psi, cpuonly=cpuonly, downsample=4)[0], seed=gen_seed)
        facelist = data.FaceList(faces)
        round.input.faces = facelist
        round.input.in_indices = []
        round.input.out_indices = list(range(len(round.input.faces.faces)))
        params = self.save_params()
        self.set_round_data(round)
        self.restore_params(params)
        self.set_modified(True)


    def onFileNew(self):
        self.filepath = 'Untitled.search'
        self.reset_to_defaults()
        self.add_first_round_stub()
        self.recreate_round_list()
        self.load_params_from_data()
        self.set_modified(False)


    def onFileNewFromModel(self):
        path, ret = QtWidgets.QFileDialog.getOpenFileName(self,
            'Open Model file', '', 'Model files (*.model)')
        if path == '':
            return False
        with h5py.File(path, 'r') as modelfile:
            model = data.Model.fromhdf5(modelfile)
        iter = data.SearchIteration(method='SVI_ACCUM_ADAM_InOut',
            input=data.SearchIterationData_SVI_ACCUM_ADAM_InOut(),
            in_params=model.pyroparams)
        iterlist = data.SearchIterationList(iterations=[iter])
        self.reset_to_defaults()
        self.search.iterations = iterlist
        self.recreate_round_list()
        self.set_modified(True)
        return True


    def onFileOpen(self):
        path, ret = QtWidgets.QFileDialog.getOpenFileName(self,
            'Open Search file', '', 'Search files (*.search)')
        if path == '':
            return False
        self.load_search(path)
        self.filepath = path
        self.set_modified(False)
        return True


    def onFileSave(self):
        if self.filepath == None:
            return self.onFileSaveAs()
        with h5py.File(self.filepath, 'w') as outfile:
            self.search.tohdf5(outfile)
        self.set_modified(False)
        return True


    def onFileSaveAs(self):
        path, ret = QtWidgets.QFileDialog.getSaveFileName(self,
            'Save Search file', self.filepath, 'Search files (*.search)')
        if path == '':
            return False
        if not path.endswith('.search'):
            path = path + '.search'
        with h5py.File(path, 'w') as outfile:
            self.search.tohdf5(outfile)
        self.filepath = path
        self.set_modified(False)
        return True


    def closeEvent(self, event):
        if self.handle_exit_with_unsaved_changes():
            event.accept()
        else:
            event.ignore()


    def onFileQuit(self):
        self.close()        # Will trigger closeEvent above and thus the save code


    def onFaceExport(self):
        if self.detail_face == None:
            return False
        path, ret = QtWidgets.QFileDialog.getSaveFileName(self,
            'Save Face file', '', 'Face files (*.face)')
        if path == '':
            return False
        if not path.endswith('.face'):
            path = path + '.face'
        with h5py.File(path, 'w') as outfile:
            self.detail_face.tohdf5(outfile)
        return True


    def onModelExport(self):
        round = self.get_current_round()
        if round == None:
            return False
        dlg = SaveModelDlg()
        if dlg.exec_() != 1:
            return
        path, name, desc = dlg.getValue()
        if path == '':
            return False
        if not path.endswith('.model'):
            path = path + '.model'
        model = data.Model(name=name, desc=desc, modeltype=self.search.modeltype,
            gentype=self.search.gentype, pyroparams=round.in_params)
        with h5py.File(path, 'w') as outfile:
            model.tohdf5(outfile)
        return True


    # TODO: Warning about breaking existing search
    def onSetGood(self):
        for item in self.bad_images.selectedItems():
            self.swap_image_class(togood=True, gui_index=self.bad_images.row(item))


    # TODO: Warning about breaking existing search
    def onSetBad(self):
        for item in self.good_images.selectedItems():
            self.swap_image_class(togood=False, gui_index=self.good_images.row(item))


    def onGoodClicked(self):
        face = self.get_current_face(good=True)
        if face == None:
            self.image_detail.clear()
        self.set_image_detail(face)
        self.update_ui_enabled_elements()


    def onBadClicked(self):
        face = self.get_current_face(good=False)
        if face == None:
            self.image_detail.clear()
        self.set_image_detail(face)
        self.update_ui_enabled_elements()


    def onRoundChange(self):
        round = self.get_current_round()
        if round == None:
            return
        self.set_round_data(round)
        self.update_ui_enabled_elements()


    # Saves work if requested. Returns True if the exit should proceed, or
    # False if it should be aborted.
    def handle_exit_with_unsaved_changes(self):
        if not self.modified_from_disk:
            return True
        response = QtWidgets.QMessageBox.question(self, "Save before exit?",
            "You have unsaved changes. Do you wish to save them before exiting?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel)
        if response == QtWidgets.QMessageBox.No:
            return True
        elif response == QtWidgets.QMessageBox.Yes:
            ret = self.onFileSaveAs()
            if ret == True:
                return True
        return False


    def update_ui_enabled_elements(self):
        if not torch.cuda.is_available():
            self.generator_cuda.setChecked(False)
            self.search_cuda.setChecked(False)
        self.generator_cuda.setEnabled(torch.cuda.is_available())
        self.search_cuda.setEnabled(torch.cuda.is_available())
        self.search_go.setEnabled(self.good_images.count() > 0 and self.bad_images.count() > 0)
        self.set_good.setEnabled(len(self.bad_images.selectedItems()) > 0)
        self.set_bad.setEnabled(len(self.good_images.selectedItems()) > 0)
        self.actionExportFace.setEnabled(self.detail_face != None)
        self.actionExportModel.setEnabled(self.get_current_round() != None and
            self.get_current_round().in_params.paramstore != b'')
        self.actionSave.setEnabled(len(self.search.iterations.iterations) > 1
            or self.good_images.count() > 0 or self.bad_images.count() > 0)
        self.actionSaveAs.setEnabled(self.actionSave.isEnabled())


    def set_modified(self, modified):
        # Mark as modified relative to the disk copy, and update the window
        # title accordingly.
        self.modified_from_disk = modified
        self.setWindowTitle('Model Search - {}{}'.format(self.filepath, '*' if modified else ''))


    def load_search(self, path):
        with h5py.File(path, 'r') as infile:
            self.reset_to_defaults()
            self.search = data.Search.fromhdf5(infile)
        for i, round in enumerate(self.search.iterations.iterations, 1):
            self.list_rounds.addItem('Round {}'.format(i))
        self.list_rounds.setCurrentRow(self.list_rounds.count()-1)
        self.filepath = path
        self.load_params_from_data()


    def recreate_round_list(self):
        self.list_rounds.clear()
        for i, round in enumerate(self.search.iterations.iterations, 1):
            self.list_rounds.addItem('Round {}'.format(i))
        self.list_rounds.setCurrentRow(self.list_rounds.count()-1)
        self.onRoundChange()


    def get_current_round(self):
        current_row = self.list_rounds.currentRow()
        if current_row == -1:
            return None
        return self.search.iterations.iterations[current_row]


    def get_previous_round(self):
        current_row = self.list_rounds.currentRow()
        if current_row < 1:
            return None
        return self.search.iterations.iterations[current_row-1]


    def get_current_face(self, good):
        round = self.get_current_round()
        if round == None:
            return None
        imagelist = self.good_images if good else self.bad_images
        imagelist_index = imagelist.currentRow()
        if imagelist_index == -1:
            return None
        assert round.method == 'SVI_ACCUM_ADAM_InOut'
        data_indices = round.input.in_indices if good else round.input.out_indices
        face = round.input.faces.faces[data_indices[imagelist_index]]
        return face


    def set_image_detail(self, face):
        pixmap = QPixmap()
        if face.thumbnail.width > 0:
            pixmap.loadFromData(face.thumbnail.data)
        else:
            pixmap.load('avatar.png')
        self.detail_face = face
        self.image_detail.setPixmap(pixmap)


    def add_image_to_gui(self, isgood, face):
        pixmap = QPixmap()
        if face.thumbnail.width > 0:
            pixmap.loadFromData(face.thumbnail.data)
        else:
            pixmap.load('avatar.png')
        image_item = QtWidgets.QListWidgetItem(QIcon(pixmap), face.name)
        listwidget = self.good_images if isgood else self.bad_images
        listwidget.addItem(image_item)


    def swap_image_class(self, togood, gui_index):
        # Moves the image identified by gui_index between classes. From good to
        # bad if togood=False, else from bad to good. Updates both the gui and
        # the data structure.
        current_round = self.get_current_round()
        if current_round == None:
            return

        from_data_indices = current_round.input.out_indices if togood else current_round.input.in_indices
        to_data_indices = current_round.input.in_indices if togood else current_round.input.out_indices
        from_images = self.bad_images if togood else self.good_images
        data_index = from_data_indices[gui_index]

        from_data_indices.remove(data_index)
        to_data_indices.append(data_index)

        from_images.takeItem(gui_index)
        face = current_round.input.faces.faces[data_index]
        self.add_image_to_gui(togood, face)


    def set_round_images(self, round):
        self.good_images.clear()
        for good_i in round.input.in_indices:
            self.add_image_to_gui(True, round.input.faces.faces[good_i])
        self.bad_images.clear()
        for bad_i in round.input.out_indices:
            self.add_image_to_gui(False, round.input.faces.faces[bad_i])


    def set_round_data(self, round):
        if round.method == 'SVI_ACCUM_ADAM_InOut':
            self.search_learningrate.setValue(round.input.lr)
            self.search_beta1.setValue(round.input.beta1)
            self.search_beta2.setValue(round.input.beta2)
            self.search_iterations.setValue(round.input.niter)
            self.set_round_images(round)
        else:
            assert False
            # TODO: Dialog box -- this method not supported


    # Reinitialise everything to defaults
    def reset_to_defaults(self):
        # Reset UI
        self.good_images.clear()
        self.bad_images.clear()
        self.image_detail.clear()
        self.detail_face = None
        self.global_description.setPlainText('')
        self.global_name.setPlainText('')
        self.set_default_params()

        # Reset data
        self.search = data.Search()
        self.list_rounds.clear()
        self.search.modeltype = 'CdV'
        self.search.gentype = 'StyleGAN'

        # Copy the UI defaults to data
        self.send_params_to_data()


    # Add a Round 1 stub from which to start the search procedure.
    def add_first_round_stub(self):
        input = data.SearchIterationData_SVI_ACCUM_ADAM_InOut(
            lr=self.search_learningrate.value(),
            beta1=self.search_beta1.value(),
            beta2=self.search_beta2.value(),
            niter=self.search_iterations.value())
        searchiter = data.SearchIteration(method='SVI_ACCUM_ADAM_InOut', input=input)
        searchiters = data.SearchIterationList([searchiter])
        self.search = data.Search(iterations=searchiters)


    # Set hyperparameters to defaults
    def set_default_params(self):
        self.generator_psi.setValue(0.7)
        self.generator_seed.setValue(27182818)
        self.sample_count.setValue(64)
        self.sample_seed.setValue(27182818)
        self.search_beta1.setValue(0.9)
        self.search_beta2.setValue(0.999)
        self.search_iterations.setValue(500)
        self.search_learningrate.setValue(0.005)


    # Update the internal data structure (self.search) with the current UI
    # settings of parameters.
    def send_params_to_data(self):
        if self.protect_data:
            return
        if self.search == None:
            return
        self.search.name = self.global_name.toPlainText()
        self.search.desc = self.global_description.toPlainText()

        round = self.get_current_round()
        if round == None:
            return
        round.method = 'SVI_ACCUM_ADAM_InOut'
        assert isinstance(round.input, data.SearchIterationData_SVI_ACCUM_ADAM_InOut)
        input = round.input
        input.lr = self.search_learningrate.value()
        input.beta1 = self.search_beta1.value()
        input.beta2 = self.search_beta2.value()
        input.niter = self.search_iterations.value()
        # Note we do not update input.faces, input.in_indices, or
        # input.out_indices, as these are handled by doSample and swap_image_class.


    # Update the UI with the parameter settings in the internal data structure
    # (self.search).
    def load_params_from_data(self):
        # Semaphore to prevent self.save_params_to_data acting on every UI
        # element update in this function and thereby undoing most changes.
        self.protect_data = True

        if self.search == None:
            return
        self.global_name.setPlainText(self.search.name)
        self.global_description.setPlainText(self.search.desc)

        round = self.get_current_round()
        if round == None:
            return
        assert round.method == 'SVI_ACCUM_ADAM_InOut'
        assert isinstance(round.input, data.SearchIterationData_SVI_ACCUM_ADAM_InOut)

        input = round.input
        self.search_learningrate.setValue(input.lr)
        self.search_beta1.setValue(input.beta1)
        self.search_beta2.setValue(input.beta2)
        self.search_iterations.setValue(input.niter)
        # Note we do not update the UI images here
        self.protect_data = False


    def save_params(self):
        return {
            'global_name': self.global_name.toPlainText(),
            'global_description': self.global_description.toPlainText(),
            'generator_psi': self.generator_psi.value(),
            'generator_seed': self.generator_seed.value(),
            'sample_count': self.sample_count.value(),
            'sample_seed': self.sample_seed.value(),
            'search_beta1': self.search_beta1.value(),
            'search_beta2': self.search_beta2.value(),
            'search_iterations': self.search_iterations.value(),
            'search_learningrate': self.search_learningrate.value()}


    def restore_params(self, params):
        self.global_name.setPlainText(params['global_name'])
        self.global_description.setPlainText(params['global_description'])
        self.generator_psi.setValue(params['generator_psi'])
        self.generator_seed.setValue(params['generator_seed'])
        self.sample_count.setValue(params['sample_count'])
        self.sample_seed.setValue(params['sample_seed'])
        self.search_beta1.setValue(params['search_beta1'])
        self.search_beta2.setValue(params['search_beta2'])
        self.search_iterations.setValue(params['search_iterations'])
        self.search_learningrate.setValue(params['search_learningrate'])




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
