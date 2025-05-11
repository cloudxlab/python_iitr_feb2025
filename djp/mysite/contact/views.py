from django.shortcuts import render, redirect
from .forms import ContactForm

def contact_us(request):
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('contact:thanks')
    else:
        form = ContactForm()
    return render(request, 'contact/contact_us.html', {'form': form})

def thanks(request):
    return render(request, 'contact/thanks.html')
