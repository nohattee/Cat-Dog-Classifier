import React from "react";

export default function SidebarDemo(props) {
  let cat_prob = Math.round(props.prob[0] * 1000) / 1000;
  let dog_prob = Math.round(props.prob[1] * 1000) / 1000;
  let result = "";
  if (cat_prob === dog_prob) {
    result = "I don't know";
  } else if (cat_prob > dog_prob) {
    result = "The cat is in the image";
  } else {
    result = "The dog is in the image";
  }
  return (
    <React.Fragment>
      <div className="container">
        <div className="row">
          <div className="col-6">
            <b>Cat</b>
          </div>
          <div className="col-6">
            {cat_prob}
          </div>
        </div>
        <div className="row">
          <div className="col-6">
            <b>Dog</b>
          </div>
          <div className="col-6">
            {dog_prob}
          </div>
        </div>
        <div className="row">
          <div className="col">
            {result}
          </div>
        </div>
      </div>
    </React.Fragment>
  )
}