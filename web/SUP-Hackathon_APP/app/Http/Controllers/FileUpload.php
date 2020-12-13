<?php

namespace App\Http\Controllers;
use Illuminate\Http\Request;
use App\Models\File;

class FileUpload extends Controller
{
  public function createForm(){
    return view('file-upload');
  }

  public function fileUpload(Request $req){
        $req->validate([
        'file' => 'required'
        ]);

        $fileModel = new File;
        if($req->file()) {
            $fileName = $req->file->getClientOriginalName();
            $filePath = $req->file('file')->storeAs('uploads', $fileName, 'public');
            $filesize=$req->file->getSize();
            $price=$filesize*0.000000108;
            $price=round( $price,2);
            $fileModel->name = time().'_'.$req->file->getClientOriginalName();
            $fileModel->file_path = '/storage/' . $filePath;
            return view('calc')->with('price',$price)->with('file', $fileName)->with('filesize', $filesize);
        }
   }

}

//4.3mb 1.08
//12.6mb 2.14
