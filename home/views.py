# from django.http import HttpResponse

# def img_capture(request):
#     return HttpResponse("hello")


from django.shortcuts import render

def home(request):
    return render(request, "index.html")


