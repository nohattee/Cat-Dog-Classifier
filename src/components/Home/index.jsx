import React, { useState } from "react";
import SidebarDemo from "../SidebarDemo";
import cn from "classnames";
import styles from "./home.module.css";
import axios from "axios";

export default function Home() {
  const [image, setImage] = useState("");
  const [prob, setProb] = useState([0, 0]);
  const [model, setModel] = useState("model");

  const onChange = e => {
    if (typeof(e.target.files[0]) !== "undefined") {
      const newImage = URL.createObjectURL(e.target.files[0]);
      setImage(newImage);
      const formData = new FormData();
      formData.append("img", e.target.files[0]);
      const config = {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      };
      axios.post("/predict/" + model, formData, config)
        .then((res) => {setProb(res.data[0])})
          .catch((err) => {console.log(err)});
    }
  }

  return (
    <React.Fragment>
      <div className="container h-100">
        <div className="row h-100">
          <div className={cn(styles.imageView, "col-sm-9 d-flex align-items-center bg-dark")}>
            <img className="img-fluid rounded mx-auto d-block" src={image} alt=""/>
          </div>
          <div className="col-sm-3 bg-light">
            <SidebarDemo prob={prob} />
          </div>
        </div>
      </div>
      <input className="form-control" type="file" onChange={e => onChange(e)}/>
      <select className="form-control" value={model} onChange={e => setModel(e.target.value)}>
        <option value="model">Our Model</option>
        <option value="vgg16">VGG16</option>
        <option value="resnet18">Resnet18</option>
      </select>
    </React.Fragment>
      
  );
}
