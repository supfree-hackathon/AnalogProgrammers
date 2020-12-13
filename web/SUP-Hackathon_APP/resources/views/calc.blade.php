@include('nav')
<div class="container">
    <br>
    <br>
    <br>
    <div class="container h-100 d-flex justify-content-center">
        <div class=" my-auto">
            <h3 class="alert alert-success">File Uploaded!</h3>
            <h4>Name :  {{$file}}</h4>
            <h4>Price :  {{$price}} €</h4>
            <button type="button" class="btn btn-primary btn-block">Add to Cart</button>


        </div>
    </div>
   
</div>
