import SimpleITK as sitk

class PrepImage:
    def __init__(self, image):
        self._image = image
        self._pipeline = []

    # Add a function to the pipeline, the function must take an image as the first argument and return an image as output
    def add_pipeline_function(self, func, *args, **kwargs):
        self._pipeline.append((func, args, kwargs))

    # Add a filter to the pipeline, the filter must have an Execute method that takes an image as input and returns an image as output
    def add_pipline_filter(self,filter):
        if not hasattr(filter, 'Execute'):
            raise ValueError("Provided filter does not have an Execute method")
        
        if not callable(getattr(filter, 'Execute')):
            raise ValueError("Provided filter's Execute attribute is not callable")
        
        self._pipeline.append((filter))

        
    def Execute(self):

        result = sitk.Image(self._image)
        for obj in self._pipeline:
            if len(obj) == 3:
                func, args, kwargs = obj
                result = func(result, *args, **kwargs)
            elif len(obj) == 1:
                filter = obj
                result = filter.Execute(result)

        return result   
    
